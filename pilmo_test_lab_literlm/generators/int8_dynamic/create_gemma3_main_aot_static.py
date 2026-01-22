import os
import torch
import torch.nn as nn
import math
import numpy as np
import ai_edge_torch
from ai_edge_torch.generative.layers.kv_cache import KVCacheEntry
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping

# Model Configuration (Gemma3-1B)
NUM_LAYERS = 26
KV_CACHE_LEN = 1280
MASK_LEN = 1281
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
    d = x.shape[-1]
    x_left, x_right = x[..., : d // 2], x[..., d // 2 :]
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
        h = embeddings
        output_data = {}

        for i in range(NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c = pos_emb_local_cos if is_local else pos_emb_cos
            l_rope_s = pos_emb_local_sin if is_local else pos_emb_sin
            l_mask = mask_local if is_local else mask_global

            x = rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)

            B, T, _ = qkv.shape
            ng, nh, dim = (
                block.atten_func.config.num_query_groups,
                block.atten_func.config.num_heads,
                block.atten_func.config.head_dim,
            )
            group_size = nh // ng

            # GOLDEN PATTERN: transpose(1, 2) before split for NPU fusion
            qkv = qkv.view(B, T, ng, (group_size + 2), dim).transpose(1, 2)
            q, k, v = qkv.split([group_size, 1, 1], dim=-2)

            q = q.reshape(B, nh, T, dim).transpose(1, 2)
            k = k.reshape(B, ng, T, dim).transpose(1, 2)
            v = v.reshape(B, ng, T, dim).transpose(1, 2)

            q = rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            q, k = apply_rope(q, l_rope_c, l_rope_s), apply_rope(k, l_rope_c, l_rope_s)

            k_new, v_new = k.transpose(1, 2), v.transpose(1, 2).transpose(2, 3)
            output_data[f"kv_slice_k_{i}"], output_data[f"kv_slice_v_{i}"] = (
                k_new,
                v_new,
            )

            k_cache, v_cache = kwargs[f"kv_cache_k_{i}"], kwargs[f"kv_cache_v_{i}"]
            k_full, v_full = (
                torch.cat([k_cache, k_new], dim=2),
                torch.cat([v_cache, v_new], dim=3),
            )

            q_a = q.permute(0, 2, 1, 3)
            if nh > ng:
                k_full, v_full = (
                    k_full.repeat_interleave(nh // ng, dim=1),
                    v_full.repeat_interleave(nh // ng, dim=1),
                )

            scores = (q_a @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
            if block.atten_func.config.logit_softcap:
                sc = block.atten_func.config.logit_softcap
                scores = torch.tanh(scores / sc) * sc

            scores = scores + l_mask
            probs = torch.softmax(scores.float(), dim=-1).type_as(q)
            attn_out = (
                (probs @ v_full.transpose(-2, -1)).transpose(1, 2).reshape(B, T, -1)
            )

            y = block.atten_func.output_projection(attn_out)
            h = h + rms_norm(y, block.post_attn_norm.weight)
            h = h + block.ff(h)

        h = rms_norm(h, self.final_norm.weight)
        output_data["logits"] = self.lm_head(h)
        return output_data


def main():
    output_dir = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main_int8_static"
    os.makedirs(output_dir, exist_ok=True)
    float_path = os.path.join(output_dir, "gemma3_1b_main_float.tflite")
    quant_path = os.path.join(output_dir, "gemma3_1b_main_int8.tflite")
    aot_path = os.path.join(output_dir, "gemma3_1b_main_int8_aot.tflite")

    model = Gemma3Main(decoder.Decoder(decoder.get_decoder_config_1b())).eval()

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

    print("Exporting Float Model...")
    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature("decode", model, sample_kwargs=sample_kwargs)
    edge_model = conv.convert()
    edge_model.export(float_path)

    print("Quantizing with Static INT16/8 (Mimic style)...")
    qt = quantizer.Quantizer(float_model=float_path)
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=16,
        weight_num_bits=8,
    )

    from ai_edge_quantizer.utils import tfl_interpreter_utils

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        float_path, num_samples=1
    )
    model_qsvs = qt.calibrate(test_data)

    # Inject Golden Scales for NPU Stability
    for name in model_qsvs.keys():
        if any(x in name.lower() for x in ["embeddings", "mask", "pos_emb"]):
            model_qsvs[name] = {
                "min": np.array([-32767.0], dtype=np.float32),
                "max": np.array([32767.0], dtype=np.float32),
            }

    result = qt.quantize(calibration_result=model_qsvs)
    result.export_model(quant_path, overwrite=True)

    print("AOT Compiling...")
    try:
        litert_model = litert_types.Model.create_from_bytes(result.quantized_model)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        aot_result = aot_lib.aot_compile(litert_model, config=config)
        if aot_result.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(aot_result.models_with_backend[0][1].model_bytes)
            print(f"SUCCESS: AOT Model saved at {aot_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
