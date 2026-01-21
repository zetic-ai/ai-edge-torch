import os
import torch
import torch.nn as nn
import ai_edge_torch

# Official Configuration for Gemma3-1B
VOCAB_SIZE = 262144
EMBEDDING_DIM = 1152


class Gemma3EmbedderFP32(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Natural FP32 embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor):
        # input: int32[1, T]
        # output: float32[1, T, 1152]
        return self.embedding(token_ids)


def main():
    output_dir = "./pilmo_test_lab_literlm/bin"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_embedder.tflite")

    model = Gemma3EmbedderFP32().eval()

    print("Converting Gemma3 Embedder Model (Natural FP32)...")
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

    edge_model = conv.convert()
    edge_model.export(tflite_path)

    print(f"SUCCESS: Clean FP32 Embedder Model saved at {tflite_path}")


if __name__ == "__main__":
    main()
