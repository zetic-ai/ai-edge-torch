import tensorflow as tf
import os


def inspect_signatures(file_path):
    print(f"\n--- Signatures for {file_path} ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=file_path)
        # signature_list() returns {name: {'inputs': {...}, 'outputs': {...}}}
        sigs = interpreter.get_signature_list()
        for name, details in sigs.items():
            print(f"  Signature: {name}")
            print(f"    Inputs: {details['inputs']}")
            print(f"    Outputs: {details['outputs']}")
    except Exception as e:
        print(f"  Error reading signatures: {e}")


if __name__ == "__main__":
    base_path = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model"
    )
    inspect_signatures(os.path.join(base_path, "real_prefill_decode.tflite"))
    inspect_signatures(os.path.join(base_path, "real_aux.tflite"))
    inspect_signatures(os.path.join(base_path, "real_embedding.tflite"))
