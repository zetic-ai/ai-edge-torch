import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# Gemma 3 1B Config
VOCAB_SIZE = 262144
EMBED_DIM = 1152
REPO_ID = "google/gemma-3-1b-it"


def download_and_analyze_embeddings():
    print(f"--- Downloading weights from {REPO_ID} ---")
    # Download only the necessary safetensors files (usually contains embeddings in the first shard)
    # We use snapshot_download to get the local path
    model_path = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["*.safetensors", "config.json"],
        local_files_only=False,
    )

    print(f"Model downloaded at: {model_path}")

    # Locate safetensors files
    files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
    files.sort()

    embed_weight = None

    print("Searching for embedding weights...")
    for f in files:
        f_path = os.path.join(model_path, f)
        state_dict = load_file(f_path)

        # Gemma 3 HF naming is usually 'model.embed_tokens.weight'
        if "model.embed_tokens.weight" in state_dict:
            embed_weight = state_dict["model.embed_tokens.weight"]
            print(f"Found embeddings in {f}!")
            break

    if embed_weight is None:
        print(
            "Error: Could not find 'model.embed_tokens.weight' in the downloaded files."
        )
        return None

    # Gemma 3 often uses a scaling factor on embeddings: sqrt(EMBED_DIM)
    # But for raw weight analysis, let's look at the weight itself first.
    with torch.no_grad():
        max_val = torch.max(torch.abs(embed_weight)).item()
        mean_val = torch.mean(embed_weight).item()
        std_val = torch.std(embed_weight).item()

    print("\n" + "=" * 50)
    print(f"📊 REAL EMBEDDING STATS (from HF: {REPO_ID})")
    print(f"Max Absolute Value: {max_val:.8f}")
    print(f"Mean:               {mean_val:.8f}")
    print(f"Std Dev:            {std_val:.8f}")
    print(f"Expected INT16 Scale (Max/32767): {max_val / 32767:.12f}")
    print("=" * 50)

    # Save the embedding weights to a local file for easy access in our export scripts
    save_path = "gemma3_quant_test/output/real_embed_weight.pt"
    os.makedirs("gemma3_quant_test/output", exist_ok=True)
    torch.save(embed_weight, save_path)
    print(f"Real embedding weights saved to {save_path}")

    return max_val


if __name__ == "__main__":
    download_and_analyze_embeddings()
