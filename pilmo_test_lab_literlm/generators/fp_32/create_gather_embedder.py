import torch
from torch import nn
import ai_edge_torch
import os


class GatherEmbedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, x):
        # Indexing in PyTorch often exports as GATHER
        return self.weight[x]


def create_gather_embedder():
    vocab_size = 262144
    dim = 1152
    model = GatherEmbedder(vocab_size, dim).eval()

    output_dir = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gemma3_gather_embedder.tflite")

    print(f"Exporting Gather-based embedder to {output_path}...")
    # Use a small dummy input
    dummy_input = torch.tensor([[1]], dtype=torch.int32)
    edge_model = ai_edge_torch.convert(model, (dummy_input,))
    edge_model.export(output_path)
    print("Export successful.")


if __name__ == "__main__":
    create_gather_embedder()
