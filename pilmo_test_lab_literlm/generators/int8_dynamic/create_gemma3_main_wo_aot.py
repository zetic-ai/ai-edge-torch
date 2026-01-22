import os
import torch
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_torch.generative.quantize import quant_recipes

# Model Configuration (Gemma3-1B)
NUM_LAYERS = 26
KV_CACHE_LEN = 120
HEAD_DIM = 256
EMBED_DIM = 1152


def main():
    output_dir = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main_wo_int8"
    )
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_main_wo_int8.tflite")
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")

    print("Building Official Gemma3-1B Decoder (Weight-Only INT8)...")
    config = decoder.get_decoder_config_1b()
    config.enable_hlfb = True
    model = decoder.Decoder(config).eval()

    # Signature
    tokens = torch.zeros((1, 1), dtype=torch.int32)
    input_pos = torch.zeros((1,), dtype=torch.int32)
    kv_cache = kv_utils.KVCache(
        tuple(
            kv_utils.KVCacheEntry(
                torch.zeros((1, 1, 1280, 256)), torch.zeros((1, 1, 256, 1280))
            )
            for _ in range(26)
        )
    )
    mask = torch.zeros((1, 1, 1, 1281), dtype=torch.float32)

    print("Exporting with Weight-Only INT8 Recipe...")
    quant_config = quant_recipes.full_weight_only_recipe(model.config)
    edge_model = ai_edge_torch.convert(
        model, (tokens, input_pos, kv_cache), {"mask": mask}, quant_config=quant_config
    )
    edge_model.export(tflite_path)

    print("Qualcomm NPU AOT Compilation...")
    try:
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        result = aot_lib.aot_compile(litert_model, config=config)
        if result.models_with_backend:
            with open(aot_tflite_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(f"🎉 SUCCESS! Weight-Only INT8 AOT Model saved at {aot_tflite_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
