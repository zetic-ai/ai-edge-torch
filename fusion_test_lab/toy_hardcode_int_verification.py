import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


class HardcodeInt32Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Use register_buffer to keep weights as part of the graph state
        # We use INT32 directly as per user request
        self.register_buffer("w1", torch.randint(0, 10, (512, 512), dtype=torch.int32))
        self.register_buffer("w2", torch.randint(0, 10, (512, 512), dtype=torch.int32))

    def forward(self, x):
        # x is [1, 512] INT32
        # Pure INT MatMul
        y1 = torch.mm(x, self.w1)
        y2 = torch.mm(y1, self.w2)
        return y2


def test_hardcode_fusion():
    name = "hardcode_int32_fusion"
    print(f"\n--- Testing: {name} ---")
    output_dir = "/home/pilmo/workspace/ai-edge-torch/fusion_test_lab"
    tflite_path = os.path.join(output_dir, f"{name}.tflite")
    aot_path = tflite_path.replace(".tflite", "_aot.tflite")

    model = HardcodeInt32Model().eval()
    sample_input = torch.randint(0, 10, (1, 512)).to(torch.int32)

    try:
        # 1. Convert to TFLite (No quant_config, it's already INT)
        edge_model = ai_edge_torch.convert(model, (sample_input,))
        edge_model.export(tflite_path)

        # 2. AOT Compile
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
            os.system(f"python3 analyze_io_types.py {aot_path}")
        else:
            print(f"AOT No Backend: {name}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Error: {e}")


if __name__ == "__main__":
    test_hardcode_fusion()
