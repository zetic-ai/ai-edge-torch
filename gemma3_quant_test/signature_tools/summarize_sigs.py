import sys

import tensorflow as tf


def summarize_outputs(model_path):
    print(f"\n--- Summarizing Outputs for: {model_path} ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        sigs = interpreter.get_signature_list()

        for sig_name, sig_def in sigs.items():
            print(f"\nSignature: {sig_name}")
            inputs = sig_def["inputs"]
            outputs = sig_def["outputs"]

            # If it's a list, it might be in an older format or a specific AOT representation
            if isinstance(outputs, list):
                output_names = sorted(outputs)
            elif isinstance(outputs, dict):
                output_names = sorted(list(outputs.keys()))
            else:
                print(f"  Unknown outputs format: {type(outputs)}")
                continue

            kv_slices = [n for n in output_names if "kv_slice" in n]
            other_outputs = [n for n in output_names if "kv_slice" not in n]

            print(f"  Total Outputs: {len(output_names)}")
            print(f"  KV Slice Outputs: {len(kv_slices)}")
            print(f"  Other Outputs: {other_outputs}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    summarize_outputs(sys.argv[1])
