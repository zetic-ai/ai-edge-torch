import os
import torch
import torch.nn as nn
import ai_edge_torch
from common_config import Gemma3Config
from export_utils import export_and_compile


class Gemma3EmbedderFP32(nn.Module):
    def __init__(
        self, vocab_size=Gemma3Config.VOCAB_SIZE, embedding_dim=Gemma3Config.EMBED_DIM
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor):
        # input: int32[1, T]
        # output: float32[1, T, EMBED_DIM]
        return self.embedding(token_ids)


def main():
    tflite_path = os.path.join(Gemma3Config.OUTPUT_BIN_DIR, "gemma3_1b_embedder.tflite")
    model = Gemma3EmbedderFP32().eval()

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
        sample_kwargs={
            "token_ids": torch.zeros((1, Gemma3Config.PREFILL_T), dtype=torch.int32)
        },
    )

    # Export (Embedder usually doesn't need NPU AOT acceleration)
    export_and_compile(conv, tflite_path, aot_signatures=False)


if __name__ == "__main__":
    main()
