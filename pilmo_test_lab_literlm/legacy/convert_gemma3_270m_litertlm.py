import os
import torch
from huggingface_hub import snapshot_download
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache


def main():
    repo_id = "google/gemma-3-270m-it"
    output_dir = "./output"

    # 1. Download checkpoint from Hugging Face
    print(f"Downloading weights from Hugging Face: {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id)
    print(f"Weights downloaded to: {checkpoint_dir}")

    # 2. Build Gemma 3-270M PyTorch model
    # This function internally loads safetensors from checkpoint_dir.
    print("Building PyTorch model...")
    pytorch_model = gemma3.build_model_270m(checkpoint_dir)

    # 3. Export configuration (KV Cache optimization)
    export_config = ExportConfig()
    export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    export_config.mask_as_input = True

    # 4. Configure metadata for LiteRT-LM (.litertlm) bundling
    # Apply special prompt template for Gemma 3-it (Instruction Tuned) models.
    litertlm_config = {
        "tokenizer_model_path": os.path.join(checkpoint_dir, "tokenizer.model"),
        "start_token_id": 2,  # "<bos>"
        "stop_token_ids": [1, 106],  # ["<eos>", "<end_of_turn>"]
        "user_prompt_prefix": "<start_of_turn>user\n",
        "user_prompt_suffix": "<end_of_turn>\n",
        "model_prompt_prefix": "<start_of_turn>model\n",
        "model_prompt_suffix": "<end_of_turn>\n",
        "output_format": "litertlm",  # Instructs .litertlm bundle generation
    }

    print("Starting conversion to .litertlm format...")
    os.makedirs(output_dir, exist_ok=True)

    # 5. Execute conversion
    try:
        # This function internally converts to TFLite and packages into .litertlm.
        converter.convert_to_litert(
            pytorch_model,
            output_path=output_dir,
            output_name_prefix="gemma-3-270m",
            prefill_seq_len=1024,  # Appropriate length for mobile device performance
            kv_cache_max_len=2048,
            quantize="dynamic_int8",  # Apply 8-bit dynamic quantization
            export_config=export_config,
            **litertlm_config,
        )
        print("\n" + "=" * 50)
        print(
            f"SUCCESS: Result saved at {os.path.join(output_dir, 'gemma-3-270m.litertlm')}"
        )
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAILED] Conversion error: {e}")
        # The .litertlm builder is often included in ai-edge-litert-nightly.
        if "LiteRT-LM builder" in str(e):
            print("\nAdvice: You need the nightly packages for the .litertlm builder.")
            print(
                "Try command: pip install ai-edge-torch-nightly ai-edge-litert-nightly"
            )


if __name__ == "__main__":
    main()
