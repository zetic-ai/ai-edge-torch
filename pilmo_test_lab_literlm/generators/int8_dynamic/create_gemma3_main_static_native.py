import os
import torch
import numpy as np
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from ai_edge_torch.generative.layers import kv_cache as kv_utils
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping

# Model Configuration (Gemma3-1B)
NUM_LAYERS = 26
KV_CACHE_LEN = 1280
MASK_LEN = 1281
HEAD_DIM = 256
EMBED_DIM = 1152


def main():
    output_dir = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main_native_static"
    os.makedirs(output_dir, exist_ok=True)
    float_path = os.path.join(output_dir, "gemma3_main_float.tflite")
    quant_path = os.path.join(output_dir, "gemma3_main_int8.tflite")
    aot_path = os.path.join(output_dir, "gemma3_main_int8_aot.tflite")

    print("Building Official Gemma3-1B Decoder (Direct)...")
    config = decoder.get_decoder_config_1b()
    config.enable_hlfb = True
    model = decoder.Decoder(config).eval()

    # Signature: 1-token decode
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

    print("Exporting Float TFLite...")
    # Using the standard ai_edge_torch way
    edge_model = ai_edge_torch.convert(
        model, (tokens, input_pos, kv_cache), {"mask": mask}
    )
    edge_model.export(float_path)

    print("Quantizing with Static INT16/8 + Mimic Calibration...")
    qt = quantizer.Quantizer(float_model=float_path)
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=16,
        weight_num_bits=8,
    )

    # Calibration with Dummy Data to capture keys
    from ai_edge_quantizer.utils import tfl_interpreter_utils

    test_data = tfl_interpreter_utils.create_random_normal_input_data(
        float_path, num_samples=1
    )
    model_qsvs = qt.calibrate(test_data)

    # Inject Golden Scales for Inputs/Masks to avoid splits
    for name in model_qsvs.keys():
        if any(
            x in name.lower() for x in ["tokens", "mask", "pos", "cos", "sin", "args_0"]
        ):
            model_qsvs[name] = {
                "min": np.array([-32767.0], dtype=np.float32),
                "max": np.array([32767.0], dtype=np.float32),
            }

    result = qt.quantize(calibration_result=model_qsvs)
    result.export_model(quant_path, overwrite=True)

    print("Qualcomm NPU AOT Compilation...")
    try:
        litert_model = litert_types.Model.create_from_bytes(result.quantized_model)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        aot_result = aot_lib.aot_compile(litert_model, config=config)
        if aot_result.models_with_backend:
            with open(aot_path, "wb") as f:
                f.write(aot_result.models_with_backend[0][1].model_bytes)
            print(f"🎉 SUCCESS! 1/0 INTERLEAVED AOT MODEL saved at {aot_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
