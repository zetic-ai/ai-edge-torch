import os

import numpy as np
import torch
import torch.nn as nn
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch

# Gemma 3 1B Config
OFFICIAL_MASK_SCALE = 0.0030518509447574615
TARGET_MAX = 100.0
KV_CACHE_MAX_LEN = 1280


class Gemma3MaskOfficial(nn.Module):
    """
    LOGICAL MASK GENERATOR for Gemma 3
    Supports multi-signature export for both Decode (1281) and Prefill (1408) widths.
    """

    def __init__(self, mask_val=-100.0, window_size=256):
        super().__init__()
        self.mask_val = mask_val
        self.window_size = window_size

    def forward(self, time_step: torch.Tensor, input_tokens: torch.Tensor):
        # Force graph dependency on input_tokens
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

        # Separate outputs with unique patterns
        mask_global_out = (mask_global + padding_mask) + 1.11e-7
        mask_local_out = (mask_local + padding_mask) + 2.22e-7

        return {"mask_global": mask_global_out, "mask_local": mask_local_out}


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_mask_fp32.tflite")
    quant_path = os.path.join(output_dir, "gemma3_1b_mask_a16_GOLD.tflite")
    aot_path = os.path.join(output_dir, "gemma3_1b_mask_a16_GOLD_aot.tflite")

    print("\n--- [GOLD] Exporting Multi-Signature FP32 Model ---")
    model = Gemma3MaskOfficial().eval()
    conv = ai_edge_torch._convert.converter.Converter()

    conv.add_signature(
        "decode_mask",
        model,
        sample_kwargs={
            "time_step": torch.tensor(128, dtype=torch.int32),
            "input_tokens": torch.tensor([[0]], dtype=torch.int32),
        },
    )

    conv.add_signature(
        "prefill_mask_128",
        model,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 128), dtype=torch.int32),
        },
    )

    conv.convert().export(tflite_path)

    print("\n--- [GOLD] Quantizing with Fix for Multi-Signature ADD/SELECT_V2 Ops ---")
    qt = quantizer.Quantizer(float_model=tflite_path)

    # 🎯 FIX ROOT CAUSE: ADD weight_tensor_config to handle internal constants in element-wise ops.
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

    # Re-enable targeting of logic ops for NPU connectivity
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.ADD, op_config=op_config
    )
    # qt.update_quantization_recipe(
    #     regex=".*",
    #     operation_name=qtyping.TFLOperationName.SELECT_V2,
    #     op_config=op_config,
    # )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.OUTPUT, op_config=op_config
    )

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
    }
    res = qt.calibrate(calib_data)

    # Manual Stats Injection for Official Alignment
    for sig_res in res.values():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for name, stats in sig_res.tensor_quantization_stats.items():
                if stats.min is not None and stats.max is not None:
                    try:
                        stats.min.fill(-TARGET_MAX)
                        stats.max.fill(TARGET_MAX)
                    except:
                        pass
                else:
                    stats.min = np.array([-TARGET_MAX], dtype=np.float32)
                    stats.max = np.array([TARGET_MAX], dtype=np.float32)

    quant_result = qt.quantize(res)
    quant_result.export_model(quant_path, overwrite=True)

    print("\n--- [GOLD] AOT Compilation (SM8750) ---")
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
            print(f"🎉 GOLD AOT MULTI-SIG MASK READY: {aot_path}")
    except Exception as e:
        print(f"❌ AOT Error: {e}")


if __name__ == "__main__":
    main()
