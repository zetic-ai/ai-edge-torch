import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes

# Official Configuration for Gemma3-1B
VOCAB_SIZE = 262144
EMBEDDING_DIM = 1152


class Gemma3Embedder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Internal embedding weight
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor):
        # input: int32[1, T]
        # output: float32[1, T, 1152]
        return self.embedding(token_ids)


def main():
    output_dir = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/embedder_int8"
    )
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_embedder_int8.tflite")

    model = Gemma3Embedder().eval()

    print("Converting Gemma3 Embedder Model (INT8 Dynamic)...")
    conv = ai_edge_torch._convert.converter.Converter()

    # 1. decode_embedder Signature
    conv.add_signature(
        "decode_embedder",
        model,
        sample_kwargs={"token_ids": torch.zeros((1, 1), dtype=torch.int32)},
    )

    # 2. prefill_embedder_128 Signature
    conv.add_signature(
        "prefill_embedder_128",
        model,
        sample_kwargs={"token_ids": torch.zeros((1, 128), dtype=torch.int32)},
    )

    # Apply INT8 Weight-Only Quantization Recipe
    # This will quantize the lookup table weights to INT8.
    quant_config = quant_recipes.full_dynamic_recipe()

    edge_model = conv.convert(quant_config=quant_config)
    edge_model.export(tflite_path)

    print(f"SUCCESS: INT8 Dynamic Embedder Model saved at {tflite_path}")


if __name__ == "__main__":
    main()
