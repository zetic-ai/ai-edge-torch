import tensorflow as tf
import os


def analyze_succeed_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    # List all subgraphs if possible
    # We can try to get the flatbuffer data directly
    with open(path, "rb") as f:
        model_content = f.read()

    # We can't easily parse flatbuffer here without generated code,
    # but we can use the TFLite Interpreter's internal methods if we are lucky.

    print(f"Signatures: {interpreter.get_signature_list()}")

    # Let's try to find 'decode_cache_update' details
    try:
        runner = interpreter.get_signature_runner("decode_cache_update")
        print("Signature 'decode_cache_update' found.")
    except:
        # Try without the 0. prefix if needed, or check the list
        sigs = interpreter.get_signature_list()
        for k in sigs.keys():
            if "decode_cache_update" in k:
                print(f"Found signature: {k}")
                runner = interpreter.get_signature_runner(k)

    # Let's look at the ops in the model.
    # TFLite usually has one main graph or multiple for signatures.
    # We'll use get_tensor_details and _get_ops_details
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()

    print(f"\nTotal Ops in main subgraph: {len(ops)}")
    for op in ops:
        print(f"OP: {op['op_name']} Inputs:{op['inputs']} Outputs:{op['outputs']}")


if __name__ == "__main__":
    analyze_succeed_model(
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model/real_aux.tflite"
    )
