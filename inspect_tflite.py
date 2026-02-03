try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None

import argparse
import os
import sys


def print_tensor_detail(detail, name=None):
    if name:
        print(f"    Name:         {name}")
    else:
        print(f"    Name:         {detail.get('name', 'N/A')}")

    print(f"    Tensor Index: {detail.get('index', 'N/A')}")
    print(f"    Shape:        {detail.get('shape', 'N/A')}")

    dtype = detail.get("dtype", "N/A")
    if hasattr(dtype, "__name__"):
        dtype_str = dtype.__name__
    else:
        dtype_str = str(dtype)
    print(f"    DType:        {dtype_str}")

    # Quantization check
    quant = detail.get("quantization_parameters") or detail.get("quantization")
    if quant:
        scales = quant.get("scales")
        zero_points = quant.get("zero_points")
        if scales is not None and len(scales) > 0:
            print(f"    Quantized:    Yes (Scale: {scales}, Zero Point: {zero_points})")
        else:
            print("    Quantized:    No")

    print("    ---")


def inspect_tflite(model_path):
    if tf is None and tflite is None:
        print("Error: Neither 'tensorflow' nor 'tflite-runtime' is installed.")
        return

    if not os.path.exists(model_path):
        print(f"Error: File not found at {model_path}")
        return

    print(f"\n{'=' * 80}")
    print(f" Inspecting TFLite Model: {model_path}")
    print(f"{'=' * 80}")

    try:
        # Prefer tensorflow if available, otherwise use tflite_runtime
        if tf is not None:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            interpreter = tflite.Interpreter(model_path=model_path)

        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Model Metadata
    try:
        # Some versions of TF might have this
        metadata = interpreter.get_metadata_list()
        if metadata:
            print("\n[Model Metadata]")
            for m in metadata:
                print(f"  {m}")
    except Exception:
        pass

    # Signature Information
    try:
        signatures = interpreter.get_signature_list()
    except AttributeError:
        signatures = {}

    if not signatures:
        print(
            "\nNo signatures found. This might be a legacy TFLite model or one without named signatures."
        )

        print("\n[Default Inputs]")
        for detail in interpreter.get_input_details():
            print_tensor_detail(detail)

        print("\n[Default Outputs]")
        for detail in interpreter.get_output_details():
            print_tensor_detail(detail)
    else:
        print(f"\nFound {len(signatures)} Signature(s): {list(signatures.keys())}")

        for sig_name in signatures:
            print(f"\n{'-' * 40}")
            print(f" Signature: '{sig_name}'")
            print(f"{'-' * 40}")

            try:
                sig_runner = interpreter.get_signature_runner(sig_name)

                # Signature Inputs
                input_details = sig_runner.get_input_details()
                print("\n  [Inputs]")
                for name, detail in input_details.items():
                    print_tensor_detail(detail, name)

                # Signature Outputs
                output_details = sig_runner.get_output_details()
                print("\n  [Outputs]")
                for name, detail in output_details.items():
                    print_tensor_detail(detail, name)
            except Exception as e:
                print(f"  Error accessing signature '{sig_name}': {e}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TFLite model details.")
    parser.add_argument("model_path", help="Path to the .tflite model file")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    inspect_tflite(args.model_path)
