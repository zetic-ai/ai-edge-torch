import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping
import os

VOCAB_SIZE = 262144
EMBED_DIM = 1152


class Gemma3Embedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

    def forward(self, token_ids):
        # Result is (B, T, EMBED_DIM)
        return self.embedding(token_ids)


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    embedder_fp32_path = os.path.join(output_dir, "gemma3_1b_embedder_fp32.tflite")
    embedder_int8_path = os.path.join(output_dir, "gemma3_1b_embedder_int8.tflite")

    print("Building Embedder model...")
    model = Gemma3Embedder().eval()

    # Export FP32
    print("Exporting FP32 Embedder...")
    sample_inputs = (torch.randint(0, VOCAB_SIZE, (1, 1), dtype=torch.int32),)
    # Use ai_edge_torch to convert
    edge_model = ai_edge_torch.convert(model, sample_inputs)
    edge_model.export(embedder_fp32_path)

    # Quantize to INT8 (Dynamic Range - Weight Only for Embedder is common)
    print("Quantizing Embedder to INT8...")
    qt = quantizer.Quantizer(float_model=embedder_fp32_path)

    # Weight-only INT8 for Embedding layer
    # Note: Embedding op in TFLite often doesn't support full static quantization
    # as effectively as FullyConnected, but INT8 weights are fine.
    qt.add_weight_only_config(
        regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED, num_bits=8
    )

    # No calibration needed for weight-only
    quant_result = qt.quantize()
    quant_result.export_model(embedder_int8_path, overwrite=True)

    print(f"✅ Success! Embedder models created in {output_dir}")


if __name__ == "__main__":
    main()
