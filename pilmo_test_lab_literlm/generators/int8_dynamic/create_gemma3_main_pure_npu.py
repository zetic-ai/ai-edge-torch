import os
import sys
import torch
import torch.nn as nn
import math
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_torch.generative.quantize import quant_recipes
from ai_edge_torch.hlfb import StableHLOCompositeBuilder

# Add parent directory to sys.path to access common files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_config import Gemma3Config


def rms_norm(x, weight, eps=1e-6, zero_centered=True):
    if weight is None:
        return x
    norm_x = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(norm_x + eps)
    if zero_centered:
        return (x_normed * (1.0 + weight)).type_as(x)
    return (x_normed * weight).type_as(x)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x_rotated = torch.cat([-x[..., d // 2 :], x[..., : d // 2]], dim=-1)
    return (x * cos) + (x_rotated * sin)


def get_pure_npu_main_class(num_layers):
    kv_k_names = [f"kv_cache_k_{i}" for i in range(num_layers)]
    kv_v_names = [f"kv_cache_v_{i}" for i in range(num_layers)]
    all_arg_names = (
        [
            "embeddings",
            "mask_global",
            "mask_local",
            "pos_emb_cos",
            "pos_emb_sin",
            "pos_emb_local_cos",
            "pos_emb_local_sin",
        ]
        + kv_k_names
        + kv_v_names
    )

    arg_list = ", ".join(all_arg_names)

    class_src = f"""
class PureNPUGemma3Main(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.num_layers = self.config.num_layers
        self.transformer_blocks = model.transformer_blocks
        self.final_norm = model.final_norm
        self.lm_head = model.lm_head

    def forward(self, {arg_list}):
        k_in_list = [{", ".join(kv_k_names)}]
        v_in_list = [{", ".join(kv_v_names)}]
        
        # 1. HLFB Marking - The "Magic Key" for Single DISPATCH_OP
        builder = StableHLOCompositeBuilder(name="ai_edge_torch.generative.npu.examples.gemma3.decoder.Transformer")
        marked = builder.mark_inputs(embeddings, mask_global, mask_local, pos_emb_cos, pos_emb_sin, pos_emb_local_cos, pos_emb_local_sin, *k_in_list, *v_in_list)
        
        h = marked[0]
        m_global, m_local = marked[1], marked[2]
        c_global, s_global = marked[3], marked[4]
        c_local, s_local = marked[5], marked[6]
        k_caches = marked[7 : 7 + self.num_layers]
        v_caches = marked[7 + self.num_layers : 7 + 2 * self.num_layers]
        
        res_slices = {{}}
        
        for i in range(self.num_layers):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c, l_rope_s = (c_local, s_local) if is_local else (c_global, s_global)
            l_mask = m_local if is_local else m_global

            # --- Reproduced Pure NPU Logic ---
            x = rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)

            B, T, _ = qkv.shape 
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

            q = apply_rope(q, l_rope_c, l_rope_s)
            k = apply_rope(k, l_rope_c, l_rope_s)

            # New KV Slices for Output (Stateless Return)
            k_new = k.transpose(1, 2)  # [B, ng, T, dim]
            v_new = v.transpose(1, 2).transpose(2, 3)  # [B, ng, dim, T]
            res_slices[f"kv_slice_k_{{i}}"] = k_new
            res_slices[f"kv_slice_v_{{i}}"] = v_new

            # Temporal Concatenation (Stateless Attention)
            # context length 1280 + current token T -> total T+1280
            k_full = torch.cat([k_caches[i], k_new], dim=2)
            v_full = torch.cat([v_caches[i], v_new], dim=3)

            # Implicit GQA Broadcasting via @ 
            q_a = q.permute(0, 2, 1, 3) # [B, nh, T, dim]
            scores = (q_a @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
            if block.atten_func.config.logit_softcap:
                sc = block.atten_func.config.logit_softcap
                scores = torch.tanh(scores / sc) * sc

            scores = scores + l_mask
            probs = torch.softmax(scores.float(), dim=-1).type_as(q)
            
            # (B, nh, T, total_T) @ (B, ng, total_T, dim) -> (B, nh, T, dim)
            attn_out = probs @ v_full.transpose(-2, -1)
            attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)

            y = block.atten_func.output_projection(attn_out)
            h = h + rms_norm(y, block.post_atten_norm.weight)
            h = h + block.ff(h)

        h = rms_norm(h, self.final_norm.weight)
        logits = self.lm_head(h)
        
        # 3. Mark All Outputs
        ordered_outs = [res_slices[f"kv_slice_k_{{j}}"] for j in range(self.num_layers)]
        ordered_outs += [res_slices[f"kv_slice_v_{{j}}"] for j in range(self.num_layers)]
        ordered_outs.append(logits)
        
        m_outs = builder.mark_outputs(*ordered_outs)
        
        final_dict = {{f"kv_slice_k_{{j}}": m_outs[j] for j in range(self.num_layers)}}
        final_dict.update({{f"kv_slice_v_{{j}}": m_outs[self.num_layers + j] for j in range(self.num_layers)}})
        final_dict["logits"] = m_outs[-1]
        
        return final_dict
"""
    local_scope = {
        "nn": nn,
        "torch": torch,
        "math": math,
        "StableHLOCompositeBuilder": StableHLOCompositeBuilder,
        "rms_norm": rms_norm,
        "apply_rope": apply_rope,
    }
    exec(class_src, globals(), local_scope)
    return local_scope["PureNPUGemma3Main"]


def main():
    output_dir = os.path.join(Gemma3Config.OUTPUT_BIN_DIR, "main_int8_pure_npu")
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_main_pure_npu.tflite")

    print("Building Official Gemma 3 1B Decoder...")
    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    official_model = decoder.Decoder(config)

    print(f"Wrapping model in Pure NPU logic with {Gemma3Config.NUM_LAYERS} layers...")
    PureNPUClass = get_pure_npu_main_class(Gemma3Config.NUM_LAYERS)
    main_mod = PureNPUClass(official_model).eval()

    print("Setting up sample inputs for export...")

    def get_sample_inputs(t_len, mask_len):
        kwargs = {
            "embeddings": torch.zeros(
                (1, t_len, Gemma3Config.EMBED_DIM), dtype=torch.float32
            ),
            "mask_global": torch.zeros((1, 1, t_len, mask_len), dtype=torch.float32),
            "mask_local": torch.zeros((1, 1, t_len, mask_len), dtype=torch.float32),
            "pos_emb_cos": torch.zeros(
                (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
            ),
            "pos_emb_sin": torch.zeros(
                (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
            ),
            "pos_emb_local_cos": torch.zeros(
                (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
            ),
            "pos_emb_local_sin": torch.zeros(
                (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
            ),
        }
        for i in range(Gemma3Config.NUM_LAYERS):
            kwargs[f"kv_cache_k_{i}"] = torch.zeros(
                (1, 1, Gemma3Config.KV_CACHE_LEN, Gemma3Config.HEAD_DIM),
                dtype=torch.float32,
            )
            kwargs[f"kv_cache_v_{i}"] = torch.zeros(
                (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.KV_CACHE_LEN),
                dtype=torch.float32,
            )
        return kwargs

    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "decode",
        main_mod,
        sample_kwargs=get_sample_inputs(1, Gemma3Config.DECODE_MASK_LEN),
    )
    conv.add_signature(
        "prefill_128", main_mod, sample_kwargs=get_sample_inputs(128, 1280 + 128)
    )

    print("Applying Quantization Recipe...")
    # Using full_weight_only_recipe (INT8 Weights, FP16 activations) as it is more stable for QNN fusion
    quant_config = quant_recipes.full_weight_only_recipe(main_mod.config)

    # Registering backend for Conversion-Time Lowering (Pattern 2)
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

    target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
    ai_edge_torch.experimental_add_compilation_backend(target)
    print("[AOT] Registered QNN SM8750 backend for conversion-time lowering.")

    print("Converting and Exporting (this might take a few minutes)...")
    edge_model = conv.convert(quant_config=quant_config)
    edge_model.export(tflite_path)

    print(f"SUCCESS: Pure NPU Model saved at {tflite_path}")


if __name__ == "__main__":
    main()
