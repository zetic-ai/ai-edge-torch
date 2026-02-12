import os

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


class Gemma3CacheUpdate(nn.Module):
    def __init__(
        self, layer_indices=range(NUM_LAYERS), kv_cache_max_len=KV_CACHE_MAX_LEN
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.kv_cache_max_len = kv_cache_max_len

    def forward(self, input_pos: torch.Tensor, **kwargs):
        results = {}
        # input_pos is expected as int32[1]
        pos_f = input_pos.float()
        z_f = torch.zeros([1], dtype=torch.float32)

        # Coordinate calculation matching original generator (Float -> Int Cast)
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

    # Signature 1: prefill_cache_update_128
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

    # Signature 2: decode_cache_update
    decode_smpl = {"input_pos": torch.tensor([0], dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        decode_smpl[f"kv_cache_k_{i}"] = torch.zeros((1, 1, KV_CACHE_MAX_LEN, HEAD_DIM))
        decode_smpl[f"kv_slice_k_{i}"] = torch.zeros((1, 1, 1, HEAD_DIM))
        decode_smpl[f"kv_cache_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, KV_CACHE_MAX_LEN))
        decode_smpl[f"kv_slice_v_{i}"] = torch.zeros((1, 1, HEAD_DIM, 1))
    conv.add_signature("decode_cache_update", model, sample_kwargs=decode_smpl)

    conv.convert().export(tflite_path)

    print("\n--- [GOLD STEP 2] Quantizing with Multi-Signature Support ---")
    qt = quantizer.Quantizer(float_model=tflite_path)

    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        skip_checks=True,
        explicit_dequantize=False,
    )

    # Recipes for pure INT16 execution
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

    # Calibration Support for both signatures
    target_scale = 0.003087578108534217
    calib_val = target_scale * 32767

    def get_calib_kwargs(T, pos):
        kw = {"input_pos": np.array([pos], dtype=np.int32)}
        for i in range(NUM_LAYERS):
            kw[f"kv_cache_k_{i}"] = np.zeros(
                (1, 1, KV_CACHE_MAX_LEN, HEAD_DIM), dtype=np.float32
            )
            kw[f"kv_cache_k_{i}"][0, 0, 0, 0] = calib_val
            kw[f"kv_cache_k_{i}"][0, 0, 0, 1] = -calib_val
            kw[f"kv_slice_k_{i}"] = np.zeros((1, 1, T, HEAD_DIM), dtype=np.float32)
            kw[f"kv_slice_k_{i}"][0, 0, 0, 0] = calib_val
            kw[f"kv_cache_v_{i}"] = np.zeros(
                (1, 1, HEAD_DIM, KV_CACHE_MAX_LEN), dtype=np.float32
            )
            kw[f"kv_cache_v_{i}"][0, 0, 0, 0] = calib_val
            kw[f"kv_cache_v_{i}"][0, 0, 0, 1] = -calib_val
            kw[f"kv_slice_v_{i}"] = np.zeros((1, 1, HEAD_DIM, T), dtype=np.float32)
            kw[f"kv_slice_v_{i}"][0, 0, 0, 0] = calib_val
        return kw

    calib_data = {
        "prefill_cache_update_128": [get_calib_kwargs(PREFILL_T, 0)],
        "decode_cache_update": [get_calib_kwargs(1, 128)],
    }
    res = qt.calibrate(calib_data)

    # Manual Override
    target_max = calib_val
    for sig_res in res.values():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for t_name, stats in sig_res.tensor_quantization_stats.items():
                try:
                    stats.min.fill(-target_max)
                    stats.max.fill(target_max)
                except:
                    pass

    quant_result = qt.quantize(res)
    quant_result.export_model(final_path, overwrite=True)
    print(f"🎉 GOLD MULTI-SIG CACHE UPDATE READY: {final_path}")

    # Verification
    import ai_edge_litert.interpreter as li_interp

    interp = li_interp.Interpreter(final_path)
    print("\n--- FINAL VERIFICATION ---")
    all_tensors = interp.get_tensor_details()

    print("\n--- QUANTIZED TENSORS (INT16) ---")
    for t in all_tensors:
        if t["dtype"] == np.int16:
            q = t.get("quantization_parameters", {})
            scales = q.get("scales", [])
            scale = scales[0] if len(scales) > 0 else 0
            if scale > 0:
                print(f"  Tensor: {t['name']:<50} | Scale: {scale:.12f}")

    print("\n--- OPERATION LIST VERIFICATION ---")
    ops = interp._get_ops_details()
    op_counts = {}
    for op in ops:
        op_name = op["op_name"]
        op_counts[op_name] = op_counts.get(op_name, 0) + 1
    for op_name, count in op_counts.items():
        print(f"  Op: {op_name:<25} | Count: {count}")

    if "DEQUANTIZE" in op_counts or "QUANTIZE" in op_counts:
        print("\n⚠️ WARNING: Found Q/DQ nodes! Check for unintended domain conversion.")
    else:
        print("\n✅ SUCCESS: Pure INT16 execution confirmed for Multi-Signature model.")


if __name__ == "__main__":
    main()
