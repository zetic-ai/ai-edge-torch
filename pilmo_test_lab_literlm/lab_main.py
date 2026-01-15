import os
import argparse
import sys

# Ensure current directory is in path to import core/registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils import download_from_hf, get_base_export_config
from registry.models import MODEL_SPECS
from ai_edge_torch.generative.utilities import converter


def run_lab(model_key, quant, out_dir, prefill_len, kv_len):
    if model_key not in MODEL_SPECS:
        print(f"Error: Model key '{model_key}' not found in registry.py")
        print(f"Available models: {list(MODEL_SPECS.keys())}")
        return

    spec = MODEL_SPECS[model_key]

    # 1. Download checkpoint
    checkpoint_dir = download_from_hf(spec["repo_id"])

    # 2. Build model and load weights
    print(f"\n[2/3] Building PyTorch model for {model_key}...")
    # build_model functions in ai-edge-torch typically take (checkpoint_dir)
    pytorch_model = spec["builder"](checkpoint_dir)

    # 3. Convert and pack as .litertlm
    print(f"\n[3/3] Exporting to .litertlm (Format: {quant})...")
    os.makedirs(out_dir, exist_ok=True)

    litertlm_meta = spec["litertlm_metadata"].copy()
    litertlm_meta.update(
        {
            "tokenizer_model_path": os.path.join(checkpoint_dir, "tokenizer.model"),
            "output_format": "litertlm",
        }
    )

    try:
        output_file = converter.convert_to_litert(
            pytorch_model,
            output_path=out_dir,
            output_name_prefix=model_key,
            prefill_seq_len=prefill_len,
            kv_cache_max_len=kv_len,
            quantize=quant,
            export_config=get_base_export_config(),
            **litertlm_meta,
        )
        print("\n" + "=" * 50)
        print(f"SUCCESS!")
        print(f"Model: {model_key}")
        print(f"File: {output_file}")
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAILED] Conversion error: {e}")
        if "LiteRT-LM builder" in str(e):
            print(
                "\nSuggestion: Please install nightly packages for .litertlm support."
            )
            print("pip install ai-edge-torch-nightly ai-edge-litert-nightly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiteRT-LM Experiment Lab")
    parser.add_argument(
        "--model", type=str, default="gemma-3-270m", help="Model key from registry"
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="dynamic_int8",
        help="dynamic_int8, dynamic_int4_block32, etc.",
    )
    parser.add_argument(
        "--out_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--prefill", type=int, default=1024, help="Max prefill seq length"
    )
    parser.add_argument("--kv", type=int, default=2048, help="Max KV cache length")

    args = parser.parse_args()
    run_lab(args.model, args.quant, args.out_dir, args.prefill, args.kv)
