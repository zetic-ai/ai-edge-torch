import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_torch.quantize.pt2e_quantizer import (
    PT2EQuantizer,
    get_symmetric_quantization_config,
)
from torchao.quantization.pt2e import quantize_pt2e as qt
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


class ToyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 512)
        self.linear2 = nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def test_static_fusion():
    name = "static_int8_fusion"
    print(f"\n--- Testing: {name} ---")
    output_dir = "/home/pilmo/workspace/ai-edge-torch/fusion_test_lab"
    tflite_path = os.path.join(output_dir, f"{name}.tflite")
    aot_path = tflite_path.replace(".tflite", "_aot.tflite")

    model = ToyLinear().eval()
    # Batch size must be 1 for some NPU configurations
    sample_input = torch.randn(1, 1, 512)

    try:
        # 1. Export Model and get GraphModule
        m = torch.export.export(model, (sample_input,)).module()

        # 2. Static Quantize via PT2E
        # is_dynamic=False means Static Quantization (INT8 Act + INT8 Weight)
        quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_dynamic=False)
        )
        m = qt.prepare_pt2e(m, quantizer)

        # Calibration (Forward pass with sample data to set observers)
        m(sample_input)

        m = qt.convert_pt2e(m, fold_quantize=True)

        # 3. Convert to TFLite
        edge_model = ai_edge_torch.convert(m, (sample_input,))
        edge_model.export(tflite_path)

        # 4. AOT Compile
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        result = aot_lib.aot_compile(litert_model, config=config)

        if result.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(f"AOT SUCCESS: {name}")
            # Analyze to see if FC has INT8 input
            os.system(f"python3 analyze_io_types.py {aot_path}")
        else:
            print(f"AOT No Backend: {name}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")


if __name__ == "__main__":
    test_static_fusion()
