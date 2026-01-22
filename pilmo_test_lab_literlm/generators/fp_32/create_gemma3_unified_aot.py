import os
import torch
import torch.nn as nn
import numpy as np
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping

# Model Configuration
NUM_LAYERS = 26
HEAD_DIM = 256
EMBED_DIM = 1152
VOCAB_SIZE = 262144


class GatherEmbedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return self.weight[x]


def create_unified_quant_aot():
    output_dir = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/unified_aot"
    )
    os.makedirs(output_dir, exist_ok=True)
    float_path = os.path.join(output_dir, "gemma3_unified_float.tflite")
    quant_path = os.path.join(output_dir, "gemma3_unified_int8.tflite")
    aot_path = os.path.join(output_dir, "gemma3_unified_int8_aot.tflite")

    # 1. Build Model with Gather-based Embedding
    print("Building Unified Gemma3-1B with Gather-based Embedding...")
    config = decoder.get_decoder_config_1b()
    config.enable_hlfb = True
    model = decoder.Decoder(config)
    # SWAP Embedding!
    model.tok_embedding = GatherEmbedding(model.tok_embedding.weight.data)
    model.eval()

    # 2. Export Float TFLite (Unified)
    # We use a simplified signature for AOT proof (fixed 1-token decode)
    tokens = torch.zeros((1, 1), dtype=torch.int32)
    input_pos = torch.zeros((1,), dtype=torch.int32)
    # KV cache shape for 1B: [1, 1, 1280, 256]
    kv_caches = []
    for _ in range(NUM_LAYERS):
        k = torch.zeros((1, 1, 1280, 256), dtype=torch.float32)
        v = torch.zeros((1, 1, 256, 1280), dtype=torch.float32)
        kv_caches.append(kv_utils.KVCacheEntry(k, v))
    kv_cache = kv_utils.KVCache(tuple(kv_caches))

    # Generic mask
    mask = torch.zeros((1, 1, 1, 1281), dtype=torch.float32)

    print("Exporting Float Unified Model...")
    edge_model = ai_edge_torch.convert(
        model, (tokens, input_pos, kv_cache), {"mask": mask}
    )
    edge_model.export(float_path)

    # 3. Static Quantization (Mimic Calibr tion)
    print("Applying Static INT16/8 Quantization...")
    qt = quantizer.Quantizer(float_model=float_path)
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=16,
        weight_num_bits=8,
    )

    # Dummy calibration data to populate keys
    calib_data = {
        "main": [
            {
                "tokens": np.ones((1, 1), dtype=np.int32),
                "input_pos": np.array([0], dtype=np.int32),
                "kv_cache_0_k": np.zeros((1, 1, 1280, 256), dtype=np.float32),
                # ... simplified for script brevity, AI Edge Quantizer will handle keys
            }
        ]
    }
    # Using random data for calibration in this proof
    from ai_edge_quantizer.utils import tfl_interpreter_utils

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        float_path, num_samples=1
    )
    model_qsvs = qt.calibrate(test_data)

    # Inject Golden Scales (Tokens)
    # Find the token input tensor name automatically or use a regex
    for name in model_qsvs.keys():
        if "tokens" in name.lower() or "args_0" in name.lower():
            model_qsvs[name] = {
                "min": np.array([-32767.0], dtype=np.float32),
                "max": np.array([32767.0], dtype=np.float32),
            }

    result = qt.quantize(calibration_result=model_qsvs)
    result.export_model(quant_path, overwrite=True)

    # 4. AOT Compilation
    print("Fusing to NPU AOT...")
    try:
        litert_model = litert_types.Model.create_from_bytes(result.quantized_model)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        aot_result = aot_lib.aot_compile(litert_model, config=config)
        if aot_result.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(aot_result.models_with_backend[0][1].model_bytes)
            print(f"🎉 SUCCESS! Unified INT8 AOT Model saved at {aot_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    create_unified_quant_aot()
