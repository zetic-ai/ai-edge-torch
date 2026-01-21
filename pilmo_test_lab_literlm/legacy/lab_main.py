import os
import argparse
import sys
import tempfile

# Ensure current directory is in path to import core/registry
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils import download_from_hf, get_base_export_config, compile_for_qnn
from registry.models import MODEL_SPECS
from ai_edge_torch.generative.utilities import converter, litertlm_builder


def run_lab(
    model_key,
    quant,
    out_dir,
    prefill_len,
    kv_len,
    compile_qnn=False,
    qnn_target_name="SM8750",
):
    if model_key not in MODEL_SPECS:
        print(f"Error: Model key '{model_key}' not found in registry.py")
        print(f"Available models: {list(MODEL_SPECS.keys())}")
        return

    spec = MODEL_SPECS[model_key]

    # 1. Download checkpoint
    checkpoint_dir = download_from_hf(spec["repo_id"])

    # 2. Build model and load weights
    print(f"\n[2/3] Building PyTorch model for {model_key}...")
    pytorch_model = spec["builder"](checkpoint_dir)

    # 3. Export to TFLite
    print(f"\n[3/3] Exporting to TFLite (Format: {quant})...")
    os.makedirs(out_dir, exist_ok=True)

    # Common LitertLM Config
    litertlm_meta = spec["litertlm_metadata"].copy()
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
    litertlm_meta.update(
        {
            "tokenizer_model_path": tokenizer_path,
        }
    )

    try:
        if compile_qnn:
            # Step-by-step for AOT
            print("\n[Step 3a] Converting to standard TFLite first...")
            with tempfile.TemporaryDirectory() as workdir:
                tflite_path = converter.convert_to_tflite(
                    pytorch_model,
                    output_path=workdir,
                    output_name_prefix=model_key,
                    prefill_seq_len=prefill_len,
                    kv_cache_max_len=kv_len,
                    quantize=quant,
                    export_config=get_base_export_config(),
                )

                # Step 3b: Compiled for QNN
                # Filename now includes the target, e.g., gemma-3-270m_dynamic_int8_SM8750_qnn.tflite
                qnn_tflite_filename = (
                    f"{model_key}_{quant}_{qnn_target_name.upper()}_qnn.tflite"
                )
                qnn_tflite_path = os.path.join(out_dir, qnn_tflite_filename)
                compile_for_qnn(
                    tflite_path, qnn_tflite_path, soc_model_name=qnn_target_name
                )
                output_file = qnn_tflite_path

                # Step 3c: Bundle as .litertlm
                print("\n[Step 3c] Attempting to bundle QNN model as .litertlm...")
                try:
                    litertlm_builder.build_litertlm(
                        tflite_model_path=qnn_tflite_path,
                        workdir=workdir,
                        output_path=out_dir,
                        context_length=kv_len,
                        **litertlm_meta,
                    )
                    output_file = qnn_tflite_path.replace(".tflite", ".litertlm")
                except Exception as b_err:
                    print(
                        f"\n[WARNING] .litertlm bundling failed, but QNN TFLite is saved: {b_err}"
                    )
                    print("You can still use the QNN TFLite model directly.")
        else:
            # Standard conversion
            output_file = converter.convert_to_litert(
                pytorch_model,
                output_path=out_dir,
                output_name_prefix=model_key,
                prefill_seq_len=prefill_len,
                kv_cache_max_len=kv_len,
                quantize=quant,
                export_config=get_base_export_config(),
                output_format="litertlm",
                **litertlm_meta,
            )

        print("\n" + "=" * 50)
        print(f"SUCCESS!")
        print(f"Model: {model_key}")
        print(f"QNN AOT: {'Enabled' if compile_qnn else 'Disabled'}")
        print(f"File: {output_file}")
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAILED] Conversion error: {e}")
        import traceback

        traceback.print_exc()


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
        "--prefill", type=int, default=128, help="Max prefill seq length"
    )
    parser.add_argument("--kv", type=int, default=1280, help="Max KV cache length")
    parser.add_argument(
        "--compile_qnn", action="store_true", help="Enable QNN AOT compilation"
    )
    parser.add_argument(
        "--qnn_target",
        type=str,
        default="SM8750",
        help="QNN target SoC (SM8750, SM8650, etc.)",
    )

    args = parser.parse_args()
    run_lab(
        args.model,
        args.quant,
        args.out_dir,
        args.prefill,
        args.kv,
        args.compile_qnn,
        args.qnn_target,
    )
