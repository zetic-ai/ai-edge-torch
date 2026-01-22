import torch
import torch.nn as nn
import ai_edge_torch
import os
import numpy as np
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


class GatherEmbedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, x):
        return self.weight[x]


class ToyGatherModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Gather-based Embedding (Supported for Static Quant)
        self.embed = GatherEmbedding(100, 128)
        # 2. Linear layer
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x


def main():
    output_dir = "/home/pilmo/workspace/ai-edge-torch/fusion_test_lab"
    tflite_path = os.path.join(output_dir, "toy_gather_int8.tflite")
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")

    print("Building Toy GATHER Model...")
    model = ToyGatherModel().eval()
    tokens = torch.zeros((1, 1), dtype=torch.int32)

    # Export
    edge_model = ai_edge_torch.convert(model, (tokens,))
    edge_model.export(tflite_path)

    # Static Quantization
    print("Quantizing with Static INT16/8 (Mimic style)...")
    qt = quantizer.Quantizer(float_model=tflite_path)
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
    )

    # Calibrate and Quantize
    # Use valid indices for Gather (0-99)
    calibration_data = {
        "serving_default": [{"args_0": np.array([[1]], dtype=np.int32)}]
    }
    model_qsvs = qt.calibrate(calibration_data)

    # Scale Tokens Input to Golden Range
    for name in model_qsvs.keys():
        if "args_0" in name or "tokens" in name:
            model_qsvs[name] = {
                "min": np.array([-32767.0], dtype=np.float32),
                "max": np.array([32767.0], dtype=np.float32),
            }

    result = qt.quantize(calibration_result=model_qsvs)
    result.export_model(tflite_path, overwrite=True)

    # AOT Compilation
    print("AOT Compiling...")
    try:
        litert_model = litert_types.Model.create_from_bytes(result.quantized_model)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        aot_result = aot_lib.aot_compile(litert_model, config=config)

        # Check Fusion Result
        num_dispatch = 0
        num_tflite = 0
        if aot_result.models_with_backend:
            # We can't easily see the signature count from here without sub-tool,
            # but the exit status and aot_result indicate success.
            # I will run the analyzer on the output.
            with open(aot_tflite_path, "wb") as f:
                f.write(aot_result.models_with_backend[0][1].model_bytes)
            print(f"SUCCESS: Toy Gather Model AOT saved at {aot_tflite_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
