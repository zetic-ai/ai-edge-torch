import argparse
import os

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    tflite = None


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
        if tf is not None:
            interpreter = tf.lite.Interpreter(model_path=model_path)
        else:
            interpreter = tflite.Interpreter(model_path=model_path)

        print("[*] Interpreter initialized.")
    except Exception as e:
        print(f"Error initializing interpreter: {e}")
        return

    # Try Signature Information (This often works without allocate_tensors)
    try:
        signatures = interpreter.get_signature_list()
        if signatures:
            print(f"\nFound {len(signatures)} Signature(s): {list(signatures.keys())}")
            for sig_name, sig_def in signatures.items():
                print(f"\n{'-' * 40}")
                print(f" Signature: '{sig_name}'")
                print(f"{'-' * 40}")
                print(f"  Inputs:  {list(sig_def['inputs'].keys())}")
                print(f"  Outputs: {list(sig_def['outputs'].keys())}")

                # Try to get details via sig_runner if possible
                try:
                    sig_runner = interpreter.get_signature_runner(sig_name)
                    print("\n  [Signature Inputs Detail]")
                    for name, detail in sig_runner.get_input_details().items():
                        print_tensor_detail(detail, name)
                    print("\n  [Signature Outputs Detail]")
                    for name, detail in sig_runner.get_output_details().items():
                        print_tensor_detail(detail, name)
                except Exception as e:
                    print(f"\n  (Could not get detailed signature info: {e})")
                    print(f"  Inputs (Raw): {sig_def['inputs']}")
                    print(f"  Outputs (Raw): {sig_def['outputs']}")
        else:
            print("\nNo signatures found.")
    except Exception as e:
        print(f"Error getting signature list: {e}")

    # Fallback to default details
    try:
        print("\n[Default Input Details]")
        for detail in interpreter.get_input_details():
            print_tensor_detail(detail)
        print("\n[Default Output Details]")
        for detail in interpreter.get_output_details():
            print_tensor_detail(detail)
    except Exception as e:
        print(f"Error getting default details: {e}")

    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect TFLite model signatures and tensors."
    )
    parser.add_argument("model_path", help="Path to the .tflite model file")
    args = parser.parse_args()
    inspect_tflite(args.model_path)
