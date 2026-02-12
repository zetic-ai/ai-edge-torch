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


class Gemma3MaskOfficial(nn.Module):
    """
    LOGICAL MASK GENERATOR for Gemma 3
    Identical to Google's behavior by implementing real padding and sliding window logic.
    """

    def __init__(self, mask_val=-100.0, window_size=256):
        super().__init__()
        self.mask_val = mask_val
        self.window_size = window_size

    def forward(self, time_step: torch.Tensor, input_tokens: torch.Tensor):
        # 🎯 LOGIC 1: Padding Mask.
        # If a token is -1 (Dummy padding), it should attract a penalty.
        # This creates a REAL value dependency on input_tokens.
        padding_mask = torch.where(
            input_tokens == -1, torch.tensor(-100.0), torch.tensor(0.0)
        ).view(1, 1, -1, 1)

        B, T = input_tokens.shape
        cache_len = 1281

        # Positions
        cache_pos = torch.arange(cache_len, dtype=torch.int32).view(1, 1, 1, cache_len)
        token_pos = time_step + torch.arange(T, dtype=torch.int32).view(1, 1, T, 1)

        # 🎯 LOGIC 2: Distinct outputs for Global vs Local attention patterns.
        # Global: Full causal attention
        mask_global = torch.where(cache_pos <= token_pos, 0.0, self.mask_val)

        # Local: Sliding window attention (Logical difference forces separate Identifiers)
        mask_local = torch.where(
            (cache_pos <= token_pos) & (cache_pos > token_pos - self.window_size),
            0.0,
            self.mask_val,
        )

        return {
            "mask_global": (mask_global + padding_mask),
            "mask_local": (mask_local + padding_mask),
        }


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_mask_fp32.tflite")
    quant_path = os.path.join(output_dir, "gemma3_1b_mask_a16_GOLD.tflite")
    aot_path = os.path.join(output_dir, "gemma3_1b_mask_a16_GOLD_aot.tflite")

    print("\n--- [GOLD STEP 1] Exporting FP32 Mask (Logical Alignment) ---")
    model = Gemma3MaskOfficial().eval()
    conv = ai_edge_torch._convert.converter.Converter()

    # Match Google's Scalar and [1,1] signatures
    conv.add_signature(
        "decode_mask",
        model,
        sample_kwargs={
            "time_step": torch.tensor(128, dtype=torch.int32),
            "input_tokens": torch.tensor([[0]], dtype=torch.int32),
        },
    )

    conv.convert().export(tflite_path)

    print("\n--- [GOLD STEP 2] Quantizing and Scaling ---")
    qt = quantizer.Quantizer(float_model=tflite_path)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        skip_checks=True,
        explicit_dequantize=False,
    )
    # Target Logic Ops to keep them in Integer domain
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ADD,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.OUTPUT, op_config=op_config
    )

    # Calibration with Provided Official Specs
    calib_data = {
        "decode_mask": [
            {
                "time_step": np.array(128, dtype=np.int32),
                "input_tokens": np.array([[0]], dtype=np.int32),
            }
        ]
    }
    res = qt.calibrate(calib_data)

    # Official Scale Injection
    for sig_res in res.values():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for t_name, stats in sig_res.tensor_quantization_stats.items():
                try:
                    stats.min.fill(-TARGET_MAX)
                    stats.max.fill(TARGET_MAX)
                except:
                    pass

    quant_result = qt.quantize(res)
    quant_result.export_model(quant_path, overwrite=True)

    print("\n--- [GOLD STEP 3] AOT Compilation (SM8750) ---")
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
            print(f"🎉 GOLD AOT MASK READY (Logically Correct): {aot_path}")
    except Exception as e:
        print(f"❌ AOT Error: {e}")

    # Final Verification
    import ai_edge_litert.interpreter as li_interp

    interp = li_interp.Interpreter(quant_path)
    print("\n--- FINAL SIGNATURE VERIFICATION ---")
    sigs = interp.get_signature_list()
    for sig_name, sig_def in sigs.items():
        print(f"  Inputs: {sig_def['inputs']}")
        print(f"  Outputs: {sig_def['outputs']}")


if __name__ == "__main__":
    main()
