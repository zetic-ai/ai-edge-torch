import os
import re

import numpy as np
import torch
import torch.nn as nn
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
NUM_LAYERS = 26
HEAD_DIM = 256
KV_CACHE_MAX_LEN = 1280
PREFILL_T = 128

# Mask Constants
MASK_TARGET_MAX = 100.0

# RoPE Constants
ROPE_SIN_SCALE = 0.000030518509447574615
ROPE_COS_SCALE = 0.00003051851308555342
ROPE_SIN_MAX = ROPE_SIN_SCALE * 32767
ROPE_COS_MAX = ROPE_COS_SCALE * 32767

# Cache Update Official Scales
OFFICIAL_K_SCALES = {
    0: 0.0004564668342936784,
    1: 0.0004709611530415714,
    2: 0.0028618809301406145,
    3: 0.00047850009286776185,
    4: 0.0005498406826518476,
    5: 0.000750240811612457,
    6: 0.0004662529972847551,
    7: 0.0004683960578404367,
    8: 0.0005459639360196888,
    9: 0.0005190225783735514,
    10: 0.00306672602891922,
    11: 0.0008118897094391286,
    12: 0.001179813058115542,
    13: 0.0006990935653448105,
    14: 0.0004324394976720214,
    15: 0.0007349216612055898,
    16: 0.0016174042830243707,
    17: 0.0011182001326233149,
    18: 0.0005745243979617953,
    19: 0.0008386948029510677,
    20: 0.0004270999925211072,
    21: 0.0005679332534782588,
    22: 0.00047895681927911937,
    23: 0.0007918818737380207,
    24: 0.000593360688071698,
    25: 0.0005190857336856425,
}

OFFICIAL_V_SCALES = {
    0: 0.003087578108534217,
    1: 0.0014759623445570469,
    2: 0.001856528571806848,
    3: 0.0013436758890748024,
    4: 0.0011547092581167817,
    5: 0.000452719017630443,
    6: 0.0011411957675591111,
    7: 0.0009230656432919204,
    8: 0.0011095652589574456,
    9: 0.0006255250191316009,
    10: 0.0007755350088700652,
    11: 0.0011290361871942878,
    12: 0.0007220287225209177,
    13: 0.0009121410548686981,
    14: 0.0013734344393014908,
    15: 0.0009238668717443943,
    16: 0.0007477190811187029,
    17: 0.0002910494222305715,
    18: 0.001250629429705441,
    19: 0.0020714763086289167,
    20: 0.0011775336461141706,
    21: 0.001410587690770626,
    22: 0.0017083308193832636,
    23: 0.0007613276247866452,
    24: 0.0009111125837080181,
    25: 0.0007595986826345325,
}

# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================


class Gemma3MaskOfficial(nn.Module):
    def __init__(self, mask_val=-100.0, window_size=256):
        super().__init__()
        self.mask_val = mask_val
        self.window_size = window_size

    def forward(self, time_step: torch.Tensor, input_tokens: torch.Tensor):
        padding_mask = torch.where(
            input_tokens == -1, torch.tensor(-100.0), torch.tensor(0.0)
        ).view(1, 1, -1, 1)
        B, T = input_tokens.shape
        total_len = KV_CACHE_MAX_LEN + T
        cache_pos = torch.arange(total_len, dtype=torch.int32).view(1, 1, 1, total_len)
        token_pos = time_step + torch.arange(T, dtype=torch.int32).view(1, 1, T, 1)
        mask_global = torch.where(cache_pos <= token_pos, 0.0, self.mask_val)
        mask_local = torch.where(
            (cache_pos <= token_pos) & (cache_pos > token_pos - self.window_size),
            0.0,
            self.mask_val,
        )
        mask_global_out = (mask_global + padding_mask) + 1.11e-7
        mask_local_out = (mask_local + padding_mask) + 2.22e-7
        return {"mask_global": mask_global_out, "mask_local": mask_local_out}


class Gemma3RoPEOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq_g = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        inv_freq_l = 1.0 / (
            1000000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        )
        self.register_buffer("inv_freq_g", inv_freq_g)
        self.register_buffer("inv_freq_l", inv_freq_l)

    def compute_branch(self, pos_reshaped, inv_freq):
        freqs = pos_reshaped * inv_freq.view(1, 1, -1)
        c, s = torch.cos(freqs), torch.sin(freqs)
        t = pos_reshaped.shape[1]
        c_4d, s_4d = c.view(1, t, 1, HEAD_DIM // 2), s.view(1, t, 1, HEAD_DIM // 2)
        s_v_neg = s_4d * -1.0
        cos_out = torch.cat([c_4d, c_4d], dim=-1).view(1, t, 1, HEAD_DIM)
        sin_out = torch.cat([s_v_neg, s_4d], dim=-1).view(1, t, 1, HEAD_DIM)
        return cos_out, sin_out

    def forward(self, input_pos: torch.Tensor):
        t = input_pos.shape[0]
        pos_reshaped = input_pos.view(1, t, 1).float()
        cos_g, sin_g = self.compute_branch(pos_reshaped, self.inv_freq_g)
        cos_l, sin_l = self.compute_branch(pos_reshaped, self.inv_freq_l)
        return {
            "pos_emb_cos": cos_g,
            "pos_emb_sin": sin_g,
            "pos_emb_local_cos": cos_l,
            "pos_emb_local_sin": sin_l,
        }


class Gemma3CacheUpdate(nn.Module):
    def __init__(self, layer_indices=range(NUM_LAYERS)):
        super().__init__()
        self.layer_indices = list(layer_indices)

    def forward(self, input_pos: torch.Tensor, **kwargs):
        results = {}
        pos_f = input_pos.float()
        z_f = torch.zeros([1], dtype=torch.float32)
        k_idx_f = torch.cat([z_f, z_f, pos_f, z_f], dim=0)
        v_idx_f = torch.cat([z_f, z_f, z_f, pos_f], dim=0)
        k_list = [
            k_idx_f[0].int(),
            k_idx_f[1].int(),
            k_idx_f[2].int(),
            k_idx_f[3].int(),
        ]
        v_list = [
            v_idx_f[0].int(),
            v_idx_f[1].int(),
            v_idx_f[2].int(),
            v_idx_f[3].int(),
        ]

        for i in self.layer_indices:
            k_cache_key = f"kv_cache_k_{i}"
            k_slice_key = f"kv_slice_k_{i}"
            if k_cache_key in kwargs and k_slice_key in kwargs:
                results[k_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[k_cache_key], kwargs[k_slice_key], k_list
                )
            v_cache_key = f"kv_cache_v_{i}"
            v_slice_key = f"kv_slice_v_{i}"
            if v_cache_key in kwargs and v_slice_key in kwargs:
                results[v_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[v_cache_key], kwargs[v_slice_key], v_list
                )
        return results


# ==============================================================================
# MAIN EXPORT & QUANTIZATION
# ==============================================================================


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_aux_fp32.tflite")
    quant_path = os.path.join(output_dir, "gemma3_1b_aux_a16_GOLD.tflite")
    aot_path = os.path.join(output_dir, "gemma3_1b_aux_a16_GOLD_aot.tflite")

    print("\n--- [AUX GOLD] Exporting Multi-Signature FP32 Model (6 Signatures) ---")
    mask_mod = Gemma3MaskOfficial().eval()
    rope_mod = Gemma3RoPEOfficial().eval()
    cache_mod = Gemma3CacheUpdate().eval()

    conv = ai_edge_torch._convert.converter.Converter()

    # 1 & 2: Mask Signatures
    conv.add_signature(
        "decode_mask",
        mask_mod,
        sample_kwargs={
            "time_step": torch.tensor(128, dtype=torch.int32),
            "input_tokens": torch.tensor([[0]], dtype=torch.int32),
        },
    )
    conv.add_signature(
        "prefill_mask_128",
        mask_mod,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 128), dtype=torch.int32),
        },
    )

    # 3 & 4: RoPE Signatures
    conv.add_signature(
        "decode_rope", rope_mod, sample_args=(torch.zeros(1, dtype=torch.int32),)
    )
    conv.add_signature(
        "prefill_rope", rope_mod, sample_args=(torch.zeros(128, dtype=torch.int32),)
    )

    # 5 & 6: Cache Update Signatures
    prefill_cache_smpl = {"input_pos": torch.tensor([0], dtype=torch.int32)}
    decode_cache_smpl = {"input_pos": torch.tensor([0], dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        prefill_cache_smpl[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM)
        )
        prefill_cache_smpl[f"kv_slice_k_{i}"] = torch.zeros((1, 1, PREFILL_T, HEAD_DIM))
        prefill_cache_smpl[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN)
        )
        prefill_cache_smpl[f"kv_slice_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, PREFILL_T))
        decode_cache_smpl[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM)
        )
        decode_cache_smpl[f"kv_slice_k_{i}"] = torch.zeros((1, 1, 1, HEAD_DIM))
        decode_cache_smpl[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN)
        )
        decode_cache_smpl[f"kv_slice_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, 1))

    conv.add_signature(
        "prefill_cache_update_128", cache_mod, sample_kwargs=prefill_cache_smpl
    )
    conv.add_signature(
        "decode_cache_update", cache_mod, sample_kwargs=decode_cache_smpl
    )

    conv.convert().export(tflite_path)

    print("\n--- [AUX GOLD] Quantizing with Official Scaling (Multi-Signature Fix) ---")
    qt = quantizer.Quantizer(float_model=tflite_path)

    # Shared Config with FIX for internal constants in ADD/SELECT_V2
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        skip_checks=True,
        explicit_dequantize=False,
    )

    # Selection for AOT: ONLY Mask subgraphs (0 and 1) will be NPU accelerated.
    # RoPE and Cache Update will be quantized (A16) but remain on CPU/DSP.
    # Enable targeting for NPU connectivity across all relevant ops
    qt.update_quantization_recipe(
        regex=".*mask.*",
        operation_name=qtyping.TFLOperationName.ADD,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*rope.*",
        operation_name=qtyping.TFLOperationName.ADD,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*cache_update.*",
        operation_name=qtyping.TFLOperationName.DYNAMIC_UPDATE_SLICE,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.OUTPUT, op_config=op_config
    )

    # Calibration Data Consoldiation
    def get_cache_calib_kwargs(T, pos):
        kw = {"input_pos": np.array([pos], dtype=np.int32)}
        for i in range(NUM_LAYERS):
            k_val, v_val = OFFICIAL_K_SCALES[i] * 32767, OFFICIAL_V_SCALES[i] * 32767
            kw[f"kv_cache_k_{i}"] = np.zeros(
                (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM), dtype=np.float32
            )
            kw[f"kv_cache_k_{i}"][0, 0, 0, 0:2] = [k_val, -k_val]
            kw[f"kv_slice_k_{i}"] = np.zeros((1, 1, T, HEAD_DIM), dtype=np.float32)
            kw[f"kv_slice_k_{i}"][0, 0, 0, 0] = k_val
            kw[f"kv_cache_v_{i}"] = np.zeros(
                (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN), dtype=np.float32
            )
            kw[f"kv_cache_v_{i}"][0, 0, 0, 0:2] = [v_val, -v_val]
            kw[f"kv_slice_v_{i}"] = np.zeros((1, 1, HEAD_DIM, T), dtype=np.float32)
            kw[f"kv_slice_v_{i}"][0, 0, 0, 0] = v_val
        return kw

    calib_data = {
        "decode_mask": [
            {
                "time_step": np.array(128, dtype=np.int32),
                "input_tokens": np.array([[0]], dtype=np.int32),
            }
        ],
        "prefill_mask_128": [
            {
                "time_step": np.array(0, dtype=np.int32),
                "input_tokens": np.zeros((1, 128), dtype=np.int32),
            }
        ],
        "decode_rope": [{"args_0": np.array([1], dtype=np.int32)}],
        "prefill_rope": [{"args_0": np.arange(128, dtype=np.int32)}],
        "decode_cache_update": [get_cache_calib_kwargs(1, 128)],
        "prefill_cache_update_128": [get_cache_calib_kwargs(PREFILL_T, 0)],
    }
    res = qt.calibrate(calib_data)

    print("\n[*] Applying Consolidated Official Scale Logic...")
    for sig_name, sig_res in res.items():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for t_name, stats in sig_res.tensor_quantization_stats.items():
                target_max = None

                # 1. Mask Scaling
                if "mask" in sig_name:
                    target_max = MASK_TARGET_MAX

                # 2. RoPE Scaling
                elif "rope" in sig_name:
                    target_max = ROPE_COS_MAX if "cos" in t_name else ROPE_SIN_MAX

                # 3. Cache Update Scaling
                elif "cache_update" in sig_name:
                    m = re.search(r"kv_(cache|slice)_(k|v)_(\d+)", t_name)
                    if m:
                        kind, idx = m.group(2), int(m.group(3))
                        scale = (
                            OFFICIAL_K_SCALES[idx]
                            if kind == "k"
                            else OFFICIAL_V_SCALES[idx]
                        )
                        target_max = scale * 32767
                    elif "input_pos" not in t_name:
                        target_max = OFFICIAL_K_SCALES[0] * 32767  # Fallback

                if target_max is not None:
                    try:
                        if stats.min is None or stats.max is None:
                            stats.min = np.array([-target_max], dtype=np.float32)
                            stats.max = np.array([target_max], dtype=np.float32)
                        else:
                            stats.min.fill(-target_max)
                            stats.max.fill(target_max)
                    except:
                        pass

    quant_result = qt.quantize(res)
    quant_result.export_model(quant_path, overwrite=True)

    print("\n--- [AUX GOLD] SELECTIVE AOT Compilation for QNN (SM8750) ---")
    print("[*] Target: Mask (Subgraph 0, 1) -> NPU")
    print("[*] Target: RoPE/Cache (Subgraph 2-5) -> CPU/DSP (Quantized)")
    try:
        litert_model = litert_types.Model.create_from_bytes(
            quant_result.quantized_model
        )
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        # We call the library directly to use subgraphs_to_compile
        aot_res = aot_lib.aot_compile(
            litert_model,
            target=target,
            subgraphs_to_compile=[
                0,
                1,
            ],  # Index 0: decode_mask, Index 1: prefill_mask_128
        )
        if aot_res.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(aot_res.models_with_backend[0][1].model_bytes)
            print(f"🎉 GOLD SELECTIVE AOT AUX MODEL READY: {aot_path}")
            print(aot_res.compilation_report())
    except Exception as e:
        print(f"❌ AOT Error: {e}")


if __name__ == "__main__":
    main()
