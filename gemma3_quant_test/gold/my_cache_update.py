import os
import re

import numpy as np
import torch
import torch.nn as nn
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils

# Gemma 3 1B Config
NUM_LAYERS = 26
HEAD_DIM = 256
KV_CACHE_MAX_LEN = 1280
PREFILL_T = 128

# ==============================================================================
# GOOGLE OFFICIAL SCALES (Extract from provided table)
# ==============================================================================
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


class Gemma3CacheUpdate(nn.Module):
    def __init__(
        self, layer_indices=range(NUM_LAYERS), kv_cache_max_len=KV_CACHE_MAX_LEN
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.kv_cache_max_len = kv_cache_max_len

    def forward(self, input_pos: torch.Tensor, **kwargs):
        results = {}
        pos_f = input_pos.float()
        z_f = torch.zeros([1], dtype=torch.float32)

        k_idx_f = torch.cat([z_f, z_f, pos_f, z_f], dim=0)
        k_idx_i = k_idx_f.int()

        v_idx_f = torch.cat([z_f, z_f, z_f, pos_f], dim=0)
        v_idx_i = v_idx_f.int()

        k_list = [k_idx_i[0], k_idx_i[1], k_idx_i[2], k_idx_i[3]]
        v_list = [v_idx_i[0], v_idx_i[1], v_idx_i[2], v_idx_i[3]]

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


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_cache_update_fp32.tflite")
    final_path = os.path.join(output_dir, "gemma3_1b_cache_update_a16_GOLD.tflite")

    print("\n--- [GOLD STEP 1] Exporting FP32 CacheUpdate (Multi-Signature) ---")
    model = Gemma3CacheUpdate().eval()
    conv = ai_edge_torch._convert.converter.Converter()

    # Prefill Signature
    prefill_smpl = {"input_pos": torch.tensor([0], dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        prefill_smpl[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM)
        )
        prefill_smpl[f"kv_slice_k_{i}"] = torch.zeros((1, 1, PREFILL_T, HEAD_DIM))
        prefill_smpl[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN)
        )
        prefill_smpl[f"kv_slice_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, PREFILL_T))
    conv.add_signature("prefill_cache_update_128", model, sample_kwargs=prefill_smpl)

    # Decode Signature
    decode_smpl = {"input_pos": torch.tensor([0], dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        decode_smpl[f"kv_cache_k_{i}"] = torch.zeros((1, 1, KV_CACHE_MAX_LEN, HEAD_DIM))
        decode_smpl[f"kv_slice_k_{i}"] = torch.zeros((1, 1, 1, HEAD_DIM))
        decode_smpl[f"kv_cache_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, KV_CACHE_MAX_LEN))
        decode_smpl[f"kv_slice_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, 1))
    conv.add_signature("decode_cache_update", model, sample_kwargs=decode_smpl)

    conv.convert().export(tflite_path)

    print("\n--- [GOLD STEP 2] Quantizing with OFFICIAL Scale Alignment ---")
    qt = quantizer.Quantizer(float_model=tflite_path)

    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        skip_checks=True,
        explicit_dequantize=False,
    )

    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.DYNAMIC_UPDATE_SLICE,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.INPUT, op_config=op_config
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.OUTPUT, op_config=op_config
    )

    # Calibration with Official-style values
    def get_calib_kwargs(T, pos):
        kw = {"input_pos": np.array([pos], dtype=np.int32)}
        for i in range(NUM_LAYERS):
            # We use OFFICIAL_K_SCALES[i] * 32767 as the max range for calibration input
            k_val = OFFICIAL_K_SCALES[i] * 32767
            v_val = OFFICIAL_V_SCALES[i] * 32767
            kw[f"kv_cache_k_{i}"] = np.zeros(
                (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM), dtype=np.float32
            )
            kw[f"kv_cache_k_{i}"][0, 0, 0, 0] = k_val
            kw[f"kv_cache_k_{i}"][0, 0, 0, 1] = -k_val
            kw[f"kv_slice_k_{i}"] = np.zeros((1, 1, T, HEAD_DIM), dtype=np.float32)
            kw[f"kv_slice_k_{i}"][0, 0, 0, 0] = k_val
            kw[f"kv_cache_v_{i}"] = np.zeros(
                (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN), dtype=np.float32
            )
            kw[f"kv_cache_v_{i}"][0, 0, 0, 0] = v_val
            kw[f"kv_cache_v_{i}"][0, 0, 0, 1] = -v_val
            kw[f"kv_slice_v_{i}"] = np.zeros((1, 1, HEAD_DIM, T), dtype=np.float32)
            kw[f"kv_slice_v_{i}"][0, 0, 0, 0] = v_val
        return kw

    calib_data = {
        "prefill_cache_update_128": [get_calib_kwargs(PREFILL_T, 0)],
        "decode_cache_update": [get_calib_kwargs(1, 128)],
    }
    res = qt.calibrate(calib_data)

    # ==========================================================================
    # 🎯 HARIAL ALIGNMENT: Manual Statistic Injection
    # ==========================================================================
    print("\n[*] Carrying out Manual Scale Alignment with Google Official Constants...")
    for sig_res in res.values():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for t_name, stats in sig_res.tensor_quantization_stats.items():
                m = re.search(r"kv_(cache|slice)_(k|v)_(\d+)", t_name)
                if m:
                    kind = m.group(2)  # 'k' or 'v'
                    idx = int(m.group(3))
                    scale = (
                        OFFICIAL_K_SCALES[idx]
                        if kind == "k"
                        else OFFICIAL_V_SCALES[idx]
                    )
                    target_max = scale * 32767
                    try:
                        stats.min.fill(-target_max)
                        stats.max.fill(target_max)
                    except:
                        pass
                elif "input_pos" not in t_name:
                    # Default for intermediate/unmatched tensors to prevent scaling noise
                    # Using a safe large range or the first layer's scale as fallback
                    try:
                        stats.min.fill(-(OFFICIAL_K_SCALES[0] * 32767))
                        stats.max.fill(OFFICIAL_K_SCALES[0] * 32767)
                    except:
                        pass

    quant_result = qt.quantize(res)
    quant_result.export_model(final_path, overwrite=True)
    print(f"🎉 GOLD OFFICIAL-ALIGNED CACHE UPDATE READY: {final_path}")

    # Final Verification
    import ai_edge_litert.interpreter as li_interp

    interp = li_interp.Interpreter(final_path)
    print("\n--- FINAL SCALE VERIFICATION (OFFICIAL MATCH) ---")
    all_tensors = interp.get_tensor_details()

    # Check specific samples to confirm alignment
    target_ids = ["kv_cache_v_3", "kv_cache_k_2", "kv_slice_k_16", "kv_cache_v_0"]
    for t in all_tensors:
        for tid in target_ids:
            if tid in t["name"]:
                q = t.get("quantization_parameters", {})
                scale = q.get("scales", [0])[0]
                print(f"  Tensor: {tid:<15} | Scale: {scale:.12f}")

    print("\n--- OPERATION LIST VERIFICATION ---")
    ops = interp._get_ops_details()
    op_counts = {}
    for op in ops:
        op_name = op["op_name"]
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
    for op_name, count in op_counts.items():
        print(f"  Op: {op_name:<25} | Count: {count}")


if __name__ == "__main__":
    main()
