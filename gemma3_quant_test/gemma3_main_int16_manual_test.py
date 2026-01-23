import os
import torch
import torch.nn as nn
import math
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

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
    x_left = x[..., : d // 2]
    x_right = x[..., d // 2 :]
    x_rotated = torch.cat([-x_right, x_left], dim=-1)
    return (x * cos) + (x_rotated * sin)


class Gemma3Main(nn.Module):
    def __init__(self, model):
        super().__init__()
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
            qkv = qkv.view(B, T, ng, (group_size + 2), dim)
            q, k, v = qkv.split([group_size, 1, 1], dim=-2)
            q = q.reshape(B, T, nh, dim)
            k = k.reshape(B, T, ng, dim)
            v = v.reshape(B, T, ng, dim)
            q = rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            q = apply_rope(q, l_rope_c, l_rope_s)
            k = apply_rope(k, l_rope_c, l_rope_s)
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
            h = h + rms_norm(y, block.post_atten_norm.weight)
            h = h + block.ff(h)
        h = rms_norm(h, self.final_norm.weight)
        output_data["logits"] = self.lm_head(h)
        return output_data


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_main_fp32_for_int16.tflite")
    int16_manual_path = os.path.join(output_dir, "gemma3_1b_main_int16_manual.tflite")
    aot_path = os.path.join(output_dir, "gemma3_1b_main_int16_manual_aot.tflite")

    # 1. Export Model
    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    main_mod = Gemma3Main(decoder.Decoder(config)).eval()

    sample_kwargs = {
        "embeddings": torch.zeros((1, 1, EMBED_DIM)),
        "mask_global": torch.zeros((1, 1, 1, MASK_LEN)),
        "mask_local": torch.zeros((1, 1, 1, MASK_LEN)),
        "pos_emb_cos": torch.zeros((1, 1, 1, HEAD_DIM)),
        "pos_emb_sin": torch.zeros((1, 1, 1, HEAD_DIM)),
        "pos_emb_local_cos": torch.zeros((1, 1, 1, HEAD_DIM)),
        "pos_emb_local_sin": torch.zeros((1, 1, 1, HEAD_DIM)),
    }
    for i in range(NUM_LAYERS):
        sample_kwargs[f"kv_cache_k_{i}"] = torch.zeros((1, 1, KV_CACHE_LEN, HEAD_DIM))
        sample_kwargs[f"kv_cache_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, KV_CACHE_LEN))

    print("[1/4] Exporting FP32 model...")
    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature("decode", main_mod, sample_kwargs=sample_kwargs)
    edge_model = conv.convert()
    edge_model.export(tflite_path)

    # 2. Manual Quantization Setup
    print("[2/4] Setting up Manual INT16 Symmetric Quantization...")
    qt = quantizer.Quantizer(float_model=tflite_path)

    # Activation Config: Symmetric INT16 (Zero Point = 0)
    act_config = qtyping.TensorQuantizationConfig(
        num_bits=16,
        symmetric=True,
        granularity=qtyping.QuantGranularity.TENSORWISE,
        dtype=qtyping.TensorDataType.INT,
    )

    # Weight Config: Symmetric INT8 (Zero Point = 0)
    weight_config = qtyping.TensorQuantizationConfig(
        num_bits=8,
        symmetric=True,
        granularity=qtyping.QuantGranularity.CHANNELWISE,
        dtype=qtyping.TensorDataType.INT,
    )

    # Create Op Config for Static Range Quantization (SRQ)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=act_config,
        weight_tensor_config=weight_config,
        compute_precision=qtyping.ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )

    # Apply to all operations
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
    )

    # 3. Calibrate and Quantize
    print("[3/4] Calibrating and Quantizing...")
    # Use dummy calibration data (since we're focusing on structure/AOT)
    calib_data = {"decode": [sample_kwargs]}
    res = qt.calibrate(calib_data)
    quant_result = qt.quantize(res)
    quant_result.export_model(int16_manual_path, overwrite=True)

    # 4. AOT Compilation
    print("[4/4] AOT Compiling for Qualcomm SM8750...")
    try:
        litert_model = litert_types.Model.create_from_bytes(
            quant_result.quantized_model
        )
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        aot_config = [litert_types.CompilationConfig(target=target)]
        aot_res = aot_lib.aot_compile(litert_model, config=aot_config)

        if aot_res.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(aot_res.models_with_backend[0][1].model_bytes)
            print(f"✅ Success! AOT model created: {aot_path}")
        else:
            print("❌ Failure: AOT produced no models_with_backend")
    except Exception as e:
        print(f"❌ AOT Error: {e}")


if __name__ == "__main__":
    main()
