import os
import torch
import torch.nn as nn
import math
import ai_edge_torch
from ai_edge_torch.generative.layers.kv_cache import KVCacheEntry, KV_LAYOUT_TRANSPOSED
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_torch.generative.quantize import quant_recipes

# Model Configuration (Gemma3-1B - Exact Reproduction)
NUM_LAYERS = 26
KV_CACHE_LEN = 1280  # Physical/Input cache length
MASK_LEN = 1281  # Attention length (Cache + New Token)
HEAD_DIM = 256
EMBED_DIM = 1152


def rms_norm(x, weight, eps=1e-6, zero_centered=True):
    if weight is None:
        return x
    norm_x = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(norm_x + eps)
    if zero_centered:
        return (x_normed * (1.0 + weight)).type_as(x)
    return (x_normed * weight).type_as(x)


def apply_rope(x, cos, sin):
    # Manual RoPE application to match external [1, 1, 1, 256] embeddings
    d = x.shape[-1]
    x_left = x[..., : d // 2]
    x_right = x[..., d // 2 :]
    x_rotated = torch.cat([-x_right, x_left], dim=-1)
    return (x * cos) + (x_rotated * sin)


class Gemma3Main(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.transformer_blocks = model.transformer_blocks
        self.final_norm = model.final_norm
        self.lm_head = model.lm_head

    def forward(
        self,
        embeddings,
        mask_global,
        mask_local,
        pos_emb_cos,
        pos_emb_sin,
        pos_emb_local_cos,
        pos_emb_local_sin,
        **kwargs,
    ):
        h = embeddings  # [1, 1, 1152]
        output_data = {}

        for i in range(NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c = pos_emb_local_cos if is_local else pos_emb_cos
            l_rope_s = pos_emb_local_sin if is_local else pos_emb_sin
            l_mask = mask_local if is_local else mask_global

            # 1. Norm & QKV
            x = rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)

            B, T, _ = qkv.shape  # T=1
            ng = block.atten_func.config.num_query_groups
            nh = block.atten_func.config.num_heads
            dim = block.atten_func.config.head_dim
            group_size = nh // ng

            qkv = qkv.view(B, T, ng, (group_size + 2), dim)
            q, k, v = qkv.split([group_size, 1, 1], dim=-2)
            q = q.reshape(B, T, nh, dim)
            k = k.reshape(B, T, ng, dim)
            v = v.reshape(B, T, ng, dim)

            q = rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            # value_norm is typically Identity for 1B

            # 2. RoPE
            q = apply_rope(q, l_rope_c, l_rope_s)
            k = apply_rope(k, l_rope_c, l_rope_s)

            # 3. New KV Slices for Output (Returned for Aux update)
            k_new = k.transpose(1, 2)  # [1, ng, 1, 256]
            v_new = v.transpose(1, 2).transpose(2, 3)  # [1, ng, 256, 1]
            output_data[f"kv_slice_k_{i}"] = k_new
            output_data[f"kv_slice_v_{i}"] = v_new

            # 4. Temporal Concatenation (Stateless Attention)
            # Main model does NOT use input_pos. It just appends 1 token to 1280 context.
            k_cache = kwargs[f"kv_cache_k_{i}"]  # [1, ng, 1280, 256]
            v_cache = kwargs[f"kv_cache_v_{i}"]  # [1, ng, 256, 1280]

            # Resulting Context Length = 1281
            k_full = torch.cat([k_cache, k_new], dim=2)
            v_full = torch.cat([v_cache, v_new], dim=3)

            # GQA Repeat if needed
            q_a = q.permute(0, 2, 1, 3)  # [B, nh, 1, dim]
            if nh > ng:
                k_full = k_full.repeat_interleave(nh // ng, dim=1)
                v_full = v_full.repeat_interleave(nh // ng, dim=1)

            # Score calculation
            scores = (q_a @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
            if block.atten_func.config.logit_softcap:
                sc = block.atten_func.config.logit_softcap
                scores = torch.tanh(scores / sc) * sc

            # Correctly aligned with 1281-length mask
            scores = scores + l_mask
            probs = torch.softmax(scores.float(), dim=-1).type_as(q)

            attn_out = probs @ v_full.transpose(-2, -1)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)

            # Output / FF
            y = block.atten_func.output_projection(attn_out)
            h = h + rms_norm(y, block.post_atten_norm.weight)
            h = h + block.ff(h)

        h = rms_norm(h, self.final_norm.weight)
        logits = self.lm_head(h)
        output_data["logits"] = logits
        return output_data


def create_main_module():
    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    model = decoder.Decoder(config)
    return Gemma3Main(model)


def main():
    output_dir = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main_int8"
    )
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_main_int8.tflite")
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")

    main_mod = create_main_module()
    main_mod.eval()

    # Matching the 58 inputs EXACTLY as per Google Signature
    # Input Cache: 1280 / Mask: 1281 / No input_pos
    sample_kwargs = {
        "embeddings": torch.zeros((1, 1, EMBED_DIM), dtype=torch.float32),
        "mask_global": torch.zeros((1, 1, 1, MASK_LEN), dtype=torch.float32),
        "mask_local": torch.zeros((1, 1, 1, MASK_LEN), dtype=torch.float32),
        "pos_emb_cos": torch.zeros((1, 1, 1, HEAD_DIM), dtype=torch.float32),
        "pos_emb_sin": torch.zeros((1, 1, 1, HEAD_DIM), dtype=torch.float32),
        "pos_emb_local_cos": torch.zeros((1, 1, 1, HEAD_DIM), dtype=torch.float32),
        "pos_emb_local_sin": torch.zeros((1, 1, 1, HEAD_DIM), dtype=torch.float32),
    }
    for i in range(NUM_LAYERS):
        sample_kwargs[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, KV_CACHE_LEN, HEAD_DIM), dtype=torch.float32
        )
        sample_kwargs[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, KV_CACHE_LEN), dtype=torch.float32
        )

    print(
        f"Exporting Gemma3-1B (26 layers) No-Pos Model (Cache 1280, Mask 1281/1408) with INT8 Dynamic Quantization..."
    )
    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature("decode", main_mod, sample_kwargs=sample_kwargs)

    # Prefill 128 Signature
    PREFILL_T = 128
    PREFILL_MASK_LEN = KV_CACHE_LEN + PREFILL_T
    prefill_kwargs = {
        "embeddings": torch.zeros((1, PREFILL_T, EMBED_DIM), dtype=torch.float32),
        "mask_global": torch.zeros(
            (1, 1, PREFILL_T, PREFILL_MASK_LEN), dtype=torch.float32
        ),
        "mask_local": torch.zeros(
            (1, 1, PREFILL_T, PREFILL_MASK_LEN), dtype=torch.float32
        ),
        "pos_emb_cos": torch.zeros((1, PREFILL_T, 1, HEAD_DIM), dtype=torch.float32),
        "pos_emb_sin": torch.zeros((1, PREFILL_T, 1, HEAD_DIM), dtype=torch.float32),
        "pos_emb_local_cos": torch.zeros(
            (1, PREFILL_T, 1, HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_local_sin": torch.zeros(
            (1, PREFILL_T, 1, HEAD_DIM), dtype=torch.float32
        ),
    }
    for i in range(NUM_LAYERS):
        prefill_kwargs[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, KV_CACHE_LEN, HEAD_DIM), dtype=torch.float32
        )
        prefill_kwargs[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, KV_CACHE_LEN), dtype=torch.float32
        )

    conv.add_signature("prefill_128", main_mod, sample_kwargs=prefill_kwargs)

    # Apply INT8 Dynamic Quantization Recipe
    quant_config = quant_recipes.full_dynamic_recipe(main_mod.config)

    edge_model = conv.convert(quant_config=quant_config)
    edge_model.export(tflite_path)

    print("AOT Compiling (Fusing to single DISPATCH_OP)...")
    try:
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        result = aot_lib.aot_compile(litert_model, config=config)
        if result.models_with_backend:
            with open(aot_tflite_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(
                f"SUCCESS: Gemma3-1B Main No-Pos INT8 Model saved at {aot_tflite_path}"
            )
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
