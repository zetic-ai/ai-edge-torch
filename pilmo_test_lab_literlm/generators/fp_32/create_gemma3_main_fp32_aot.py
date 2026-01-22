import os
import torch
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from common_models import Gemma3Main, Gemma3Config


def get_sample_kwargs(t_len, mask_len):
    kwargs = {
        "embeddings": torch.randn(
            (1, t_len, Gemma3Config.EMBED_DIM), dtype=torch.float32
        ),
        "mask_global": torch.randn((1, 1, t_len, mask_len), dtype=torch.float32),
        "mask_local": torch.randn((1, 1, t_len, mask_len), dtype=torch.float32),
        "pos_emb_cos": torch.randn(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_sin": torch.randn(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_local_cos": torch.randn(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_local_sin": torch.randn(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
    }
    for i in range(Gemma3Config.NUM_LAYERS):
        kwargs[f"kv_cache_k_{i}"] = torch.randn(
            (1, 1, Gemma3Config.KV_CACHE_LEN, Gemma3Config.HEAD_DIM),
            dtype=torch.float32,
        )
        kwargs[f"kv_cache_v_{i}"] = torch.randn(
            (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.KV_CACHE_LEN),
            dtype=torch.float32,
        )
    return kwargs


def main():
    output_dir = os.path.join(Gemma3Config.OUTPUT_BIN_DIR, "main_fp32")
    tflite_path = os.path.join(output_dir, "gemma3_1b_main_fp32.tflite")

    print("[FP32-AOT] Loading model and preparing converter...")
    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    main_mod = Gemma3Main(decoder.Decoder(config)).eval()

    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "decode",
        main_mod,
        sample_kwargs=get_sample_kwargs(1, Gemma3Config.DECODE_MASK_LEN),
    )

    # --- Pattern 2: Conversion-Time Lowering ---
    print("[FP32-AOT] Registering QNN SM8750 backend...")
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

    target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
    conv.experimental_add_compilation_backend(target)

    # NO quantization (FP32)
    print("[FP32-AOT] Converting FP32 model with AOT...")
    result = conv.convert(strict_export=False)

    if hasattr(result, "models_with_backend") and result.models_with_backend:
        print("[FP32-AOT] AOT compilation successful. Saving bytes...")
        aot_model = result.models_with_backend[0][1]
        os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
        with open(tflite_path, "wb") as f:
            f.write(aot_model.model_bytes)
        print(f"SUCCESS: FP32 AOT Model saved at {tflite_path}")
    else:
        print("FAILED: AOT did not trigger for FP32.")


if __name__ == "__main__":
    main()
