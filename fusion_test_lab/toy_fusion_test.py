import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_torch.quantize import quant_config
from ai_edge_torch.generative.quantize import quant_recipes, quant_attrs
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


class ToyAttentionInteger(nn.Module):
    def __init__(self, use_transpose=False):
        super().__init__()
        self.use_transpose = use_transpose
        self.nh = 8
        self.ng = 2
        self.dim = 64
        self.group_size = self.nh // self.ng

        # Use Embedding to process INT inputs natively from the start
        # This acts as the first layer that translates INT -> Latent space
        self.embedding = nn.Embedding(1000, 512)

        # QKV Projection
        self.qkv_projection = nn.Linear(512, (self.nh + 2 * self.ng) * self.dim)
        # Output Projection
        self.output_projection = nn.Linear(self.nh * self.dim, 512)

    def forward(self, x):
        # x is assumed to be INT32/INT64 LongTensor (Tokens)
        x = self.embedding(x)  # Native transition from Token to Latent

        B, T, _ = x.shape
        qkv = self.qkv_projection(x)

        # Pattern matching attempt
        if self.use_transpose:
            qkv = qkv.view(B, T, self.ng, (self.group_size + 2), self.dim).transpose(
                1, 2
            )
            q, k, v = qkv.split([self.group_size, 1, 1], dim=-2)
        else:
            qkv = qkv.view(B, T, self.ng, (self.group_size + 2), self.dim)
            q, k, v = qkv.split([self.group_size, 1, 1], dim=-2)

        out = q.reshape(B, T, self.nh * self.dim)
        y = self.output_projection(out)
        return y


def test_fusion(name, model, qcfg=None):
    print(f"\n--- Testing: {name} ---")
    output_dir = "/home/pilmo/workspace/ai-edge-torch/fusion_test_lab"
    tflite_path = os.path.join(output_dir, f"{name}.tflite")
    aot_path = tflite_path.replace(".tflite", "_aot.tflite")

    # Native INT32 Input (Tokens)
    sample_input = torch.randint(0, 1000, (1, 128)).to(torch.int32)

    try:
        # Convert
        edge_model = ai_edge_torch.convert(model, (sample_input,), quant_config=qcfg)
        edge_model.export(tflite_path)

        # AOT Compile
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
        print(f"Error during {name} execution: {e}")


if __name__ == "__main__":
    # 1. INT8 Dynamic with Embedding (Native INT path)
    test_fusion(
        "native_int_dyn_no_transpose",
        ToyAttentionInteger(use_transpose=False),
        quant_recipes.full_dynamic_recipe(),
    )
    test_fusion(
        "native_int_dyn_with_transpose",
        ToyAttentionInteger(use_transpose=True),
        quant_recipes.full_dynamic_recipe(),
    )

    # 2. INT8 Weight-Only with Embedding (Native INT path)
    test_fusion(
        "native_int_wo_no_transpose",
        ToyAttentionInteger(use_transpose=False),
        quant_recipes.full_weight_only_recipe(),
    )
    test_fusion(
        "native_int_wo_with_transpose",
        ToyAttentionInteger(use_transpose=True),
        quant_recipes.full_weight_only_recipe(),
    )

    # 3. FP16 with Embedding (Native INT path)
    test_fusion(
        "native_int_fp16_no_transpose",
        ToyAttentionInteger(use_transpose=False),
        quant_recipes.full_fp16_recipe(),
    )
    test_fusion(
        "native_int_fp16_with_transpose",
        ToyAttentionInteger(use_transpose=True),
        quant_recipes.full_fp16_recipe(),
    )
