import math
import os

import torch
import torch.nn as nn
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder

# work
# ==============================================================================
# MODEL CONFIG
# ==============================================================================
NUM_LAYERS = 26
KV_CACHE_LEN = 1280
PREFILL_T = 128
HEAD_DIM = 256
EMBED_DIM = 1152


# ==============================================================================
# BASE TRANSFORMER LOGIC
# ==============================================================================
class Gemma3Base(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer_blocks = model.transformer_blocks

    def run_transformer(
        self,
        h,
        mask_global,
        mask_local,
        pos_emb_cos,
        pos_emb_sin,
        pos_emb_local_cos,
        pos_emb_local_sin,
        **kwargs,
    ):
        output_data = {}
        for i in range(NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c = pos_emb_local_cos if is_local else pos_emb_cos
            l_rope_s = pos_emb_local_sin if is_local else pos_emb_sin
            l_mask = mask_local if is_local else mask_global

            x = self.rms_norm(h, block.pre_atten_norm.weight)
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
            q, k, v = (
                q.reshape(B, T, nh, dim),
                k.reshape(B, T, ng, dim),
                v.reshape(B, T, ng, dim),
            )

            q = self.rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = self.rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            q, k = (
                self.apply_rope(q, l_rope_c, l_rope_s),
                self.apply_rope(k, l_rope_c, l_rope_s),
            )

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
            h = h + self.rms_norm(y, block.post_atten_norm.weight)
            h = h + block.ff(h)
        return h, output_data

    def rms_norm(self, x, weight, eps=1e-6, zero_centered=True):
        if weight is None:
            return x
        norm_x = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm_x + eps)
        if zero_centered:
            return (x_normed * (1.0 + weight)).type_as(x)
        return (x_normed * weight).type_as(x)

    def apply_rope(self, x, cos, sin):
        d = x.shape[-1]
        x_left, x_right = x[..., : d // 2], x[..., d // 2 :]
        x_rotated = torch.cat([-x_right, x_left], dim=-1)
        return (x * cos) + (x_rotated * sin)


# ==============================================================================
# PREFILL vs DECODE MODULES (PHYSICAL ISOLATION)
# ==============================================================================
class Gemma3Prefill(Gemma3Base):
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
        # Prefill strictly returns KV slices only
        _, output_data = self.run_transformer(
            embeddings,
            mask_global,
            mask_local,
            pos_emb_cos,
            pos_emb_sin,
            pos_emb_local_cos,
            pos_emb_local_sin,
            **kwargs,
        )
        return output_data


class Gemma3Decode(Gemma3Base):
    def __init__(self, model):
        super().__init__(model)
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
        # Decode returns KV slices + Logits
        h, output_data = self.run_transformer(
            embeddings,
            mask_global,
            mask_local,
            pos_emb_cos,
            pos_emb_sin,
            pos_emb_local_cos,
            pos_emb_local_sin,
            **kwargs,
        )
        h = self.rms_norm(h, self.final_norm.weight)
        output_data["logits"] = self.lm_head(h)
        return output_data


# ==============================================================================
# SAMPLES & MAIN
# ==============================================================================
# ==============================================================================
# SAMPLES & MAIN
# ==============================================================================
SCALES = {
    "kv_cache_k": [
        0.0004564668342936784,
        0.0004709611530415714,
        0.0028618809301406145,
        0.00047850009286776185,
        0.0005498406826518476,
        0.000750240811612457,
        0.0004662529972847551,
        0.0004683960578404367,
        0.0005459639360196888,
        0.0005190225783735514,
        0.00306672602891922,
        0.0008118897094391286,
        0.001179813058115542,
        0.0006990935653448105,
        0.0004324394976720214,
        0.0007349216612055898,
        0.0016174042830243707,
        0.0011182001326233149,
        0.0005745243979617953,
        0.0008386948029510677,
        0.0004270999925211072,
        0.0005679332534782588,
        0.00047895681927911937,
        0.0007918818737380207,
        0.000593360688071698,
        0.0005190857336856425,
    ],
    "kv_cache_v": [
        0.003087578108534217,
        0.0014759623445570469,
        0.001856528571806848,
        0.0013436758890748024,
        0.0011547092581167817,
        0.000452719017630443,
        0.0011411957675591111,
        0.0009230656432919204,
        0.0011095652589574456,
        0.0006255250191316009,
        0.0007755350088700652,
        0.0011290361871942878,
        0.0007220287225209177,
        0.0009121410548686981,
        0.0013734344393014908,
        0.0009238668717443943,
        0.0007477190811187029,
        0.0002910494222305715,
        0.001250629429705441,
        0.0020714763086289167,
        0.0011775336461141706,
        0.001410587690770626,
        0.0017083308193832636,
        0.0007613276247866452,
        0.0009111125837080181,
        0.0007595986826345325,
    ],
    "pos_emb_cos": 0.00003051851308555342,
    "pos_emb_sin": 0.000030518509447574615,
    "pos_emb_local_cos": 0.00003051851308555342,
    "pos_emb_local_sin": 0.000030518509447574615,
    "mask_global": 0.0030518509447574615,
    "mask_local": 0.0030518509447574615,
    "embeddings": 0.000022609741790802218,
}


def generate_sample(T, to_numpy=True, is_calib=False):
    # For symmetric 16-bit, S = MaxAbs / 32767 -> MaxAbs = S * 32767
    M = 32767.0

    if is_calib:

        def peak(shape, scale):
            # Create a tensor where only the first element reaches the max range to fix the scale
            t = torch.zeros(shape)
            t.flatten()[0] = scale * M
            return t

        sample = {
            "embeddings": peak((1, T, EMBED_DIM), SCALES["embeddings"]),
            "mask_global": peak((1, 1, T, KV_CACHE_LEN + T), SCALES["mask_global"]),
            "mask_local": peak((1, 1, T, KV_CACHE_LEN + T), SCALES["mask_local"]),
            "pos_emb_cos": peak((1, T, 1, HEAD_DIM), SCALES["pos_emb_cos"]),
            "pos_emb_sin": peak((1, T, 1, HEAD_DIM), SCALES["pos_emb_sin"]),
            "pos_emb_local_cos": peak((1, T, 1, HEAD_DIM), SCALES["pos_emb_local_cos"]),
            "pos_emb_local_sin": peak((1, T, 1, HEAD_DIM), SCALES["pos_emb_local_sin"]),
        }
        for i in range(NUM_LAYERS):
            sample[f"kv_cache_k_{i}"] = peak(
                (1, 1, KV_CACHE_LEN, HEAD_DIM), SCALES["kv_cache_k"][i]
            )
            sample[f"kv_cache_v_{i}"] = peak(
                (1, 1, HEAD_DIM, KV_CACHE_LEN), SCALES["kv_cache_v"][i]
            )
    else:
        # Export sample (just for shape)
        mask_len = KV_CACHE_LEN + T
        sample = {
            "embeddings": torch.randn((1, T, EMBED_DIM)),
            "mask_global": torch.zeros((1, 1, T, mask_len)),
            "mask_local": torch.zeros((1, 1, T, mask_len)),
            "pos_emb_cos": torch.randn((1, T, 1, HEAD_DIM)),
            "pos_emb_sin": torch.randn((1, T, 1, HEAD_DIM)),
            "pos_emb_local_cos": torch.randn((1, T, 1, HEAD_DIM)),
            "pos_emb_local_sin": torch.randn((1, T, 1, HEAD_DIM)),
        }
        for i in range(NUM_LAYERS):
            sample[f"kv_cache_k_{i}"] = (
                torch.randn((1, 1, KV_CACHE_LEN, HEAD_DIM)) * 0.1
            )
            sample[f"kv_cache_v_{i}"] = (
                torch.randn((1, 1, HEAD_DIM, KV_CACHE_LEN)) * 0.1
            )

    if to_numpy:
        return {
            k: v.numpy() if isinstance(v, torch.Tensor) else v
            for k, v in sample.items()
        }
    return sample


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = os.path.join(
        output_dir, "checkpoint_pilmo_optimized_main_fp32.tflite"
    )
    w4a16_path = os.path.join(
        output_dir, "checkpoint_pilmo_optimized_main_w4a16.tflite"
    )
    aot_path = os.path.join(
        output_dir, "checkpoint_pilmo_optimized_main_w4a16_aot.tflite"
    )

    print("\n--- [CORE] TRANSFORMER EXPORT (PILMO VER) ---")
    # Use the project's model builder to load real weights from Hugging Face cache.
    checkpoint_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    )
    raw_model = decoder.build_model_1b(checkpoint_path)

    prefill_mod = Gemma3Prefill(raw_model).eval()
    decode_mod = Gemma3Decode(raw_model).eval()

    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "prefill_128",
        prefill_mod,
        sample_kwargs=generate_sample(PREFILL_T, to_numpy=False),
    )
    conv.add_signature(
        "decode", decode_mod, sample_kwargs=generate_sample(1, to_numpy=False)
    )

    edge_model = conv.convert()
    edge_model.export(tflite_path)

    # Quantization
    qt = quantizer.Quantizer(float_model=tflite_path)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
    )

    calib_data = {
        "prefill_128": [generate_sample(PREFILL_T, is_calib=True)],
        "decode": [generate_sample(1, is_calib=True)],
    }
    res = qt.calibrate(calib_data)
    quant_result = qt.quantize(res)
    quant_result.export_model(w4a16_path, overwrite=True)

    print(f"🚀 AOT Compiling to {aot_path}...")
    litert_model = litert_types.Model.create_from_bytes(quant_result.quantized_model)
    target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
    aot_config = [litert_types.CompilationConfig(target=target)]
    aot_res = aot_lib.aot_compile(litert_model, config=aot_config)
    if aot_res.models_with_backend:
        with open(aot_path, "wb") as f:
            f.write(aot_res.models_with_backend[0][1].model_bytes)
        print(f"🎉 SUCCESS: Main AOT model saved at {aot_path}")


if __name__ == "__main__":
    main()
