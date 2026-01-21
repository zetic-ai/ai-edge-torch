import tensorflow as tf
import os


def summarize_tflite(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"--- Summarizing TFLite: {os.path.basename(path)} ---")
    interpreter = tf.lite.Interpreter(model_path=path)

    # Signatures
    try:
        signatures = interpreter.get_signature_list()
        print(f"Signatures: {signatures}")
    except:
        print("No signatures found.")

    # Get op details
    ops = interpreter._get_ops_details()
    print(f"\nTotal Operators: {len(ops)}")

    # To get human readable op names, we can use the main interpreter's details
    # or just look at the 'op_name' field if available in some versions,
    # but here we'll use a safer approach.

    for op in ops:
        # TFLite Interpreter ops usually have 'index', 'op_name', 'inputs', 'outputs'
        idx = op.get("index", "?")
        name = op.get("op_name", "Unknown")
        inputs = op.get("inputs", [])
        outputs = op.get("outputs", [])
        print(f"[{idx:2d}] OP: {name:25s} (In:{inputs}, Out:{outputs})")

    # Get tensor details for mapping
    tensors = interpreter.get_tensor_details()
    print("\nTensors (First 30):")
    for i, t in enumerate(tensors):
        if i < 30:
            print(f"Tensor[{t['index']:2d}]: {t['name']:30s} Shape:{t['shape']}")


if __name__ == "__main__":
    tflite_path = "./cache_update_aot_final/cache_update_fixed.tflite"
    summarize_tflite(tflite_path)
