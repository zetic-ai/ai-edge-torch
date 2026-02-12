import os

import torch
import torch.nn as nn
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch

# Gemma 3 1B Config
VOCAB_SIZE = 262144
EMBED_DIM = 1152


class Gemma3Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

    def forward(self, token_ids):
        # Result is (B, T, EMBED_DIM)
        # Ensure indices are within bounds
        token_ids = torch.maximum(token_ids, torch.tensor(0, dtype=torch.int32))
        return {"embeddings": self.tok_embedding(token_ids)}


def main():
    # Fix output directory path to be relative to the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    tflite_path = os.path.join(output_dir, "embedder_test_w8_i16io_fp32.tflite")
    quant_path = os.path.join(output_dir, "embedder_test_w8_i16io.tflite")

    print("\n--- [EMBEDDER] MULTI-SIGNATURE EXPORT (PILMO VER) ---")
    model = Gemma3Embedder().eval()

    # --- LOAD REAL WEIGHTS ---
    weight_path = os.path.join(output_dir, "real_embed_weight.pt")
    if os.path.exists(weight_path):
        print(f"Loading real embedding weights from {weight_path}...")
        real_weights = torch.load(weight_path)
        with torch.no_grad():
            model.tok_embedding.weight.copy_(real_weights)

        max_val = torch.max(torch.abs(model.tok_embedding.weight)).item()
        print(f"REAL Weight Max Abs Val = {max_val}")
    else:
        print("WARNING: real_embed_weight.pt not found. Using random weights.")
    # -------------------------

    print("Exporting FP32 Multi-Signature Embedder...")
    conv = ai_edge_torch._convert.converter.Converter()

    # Signature 1: prefill_embedder_128 [1, 128]
    conv.add_signature(
        "prefill_embedder_128",
        model,
        sample_kwargs={
            "token_ids": torch.randint(0, VOCAB_SIZE, (1, 128), dtype=torch.int32)
        },
    )

    # Signature 2: decode_embedder [1, 1]
    conv.add_signature(
        "decode_embedder",
        model,
        sample_kwargs={
            "token_ids": torch.randint(0, VOCAB_SIZE, (1, 1), dtype=torch.int32)
        },
    )

    edge_model = conv.convert()
    edge_model.export(tflite_path)

    print("Quantizing Multi-Signature Embedder to W8A16 Static...")
    qt = quantizer.Quantizer(float_model=tflite_path)

    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=16,
        weight_num_bits=8,
        weight_granularity=qtyping.QuantGranularity.CHANNELWISE,
    )

    # Calibration for both signatures
    print("Calibrating for INT16 scales (Multi-Signature)...")
    calib_data = {
        "prefill_embedder_128": [
            {
                "token_ids": torch.randint(
                    0, VOCAB_SIZE, (1, 128), dtype=torch.int32
                ).numpy()
            }
            for _ in range(5)
        ],
        "decode_embedder": [
            {
                "token_ids": torch.randint(
                    0, VOCAB_SIZE, (1, 1), dtype=torch.int32
                ).numpy()
            }
            for _ in range(5)
        ],
    }
    res = qt.calibrate(calib_data)

    quant_result = qt.quantize(res)
    quant_result.export_model(quant_path, overwrite=True)

    print(f"\n🎉 SUCCESS: Multi-Signature Embedder saved at {quant_path}")


if __name__ == "__main__":
    main()
