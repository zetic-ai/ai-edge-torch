import tensorflow as tf


def print_signatures(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    sigs = interpreter.get_signature_list()
    print(f"Signatures in {tflite_path}:")
    for sig_name, sig_def in sigs.items():
        print(f"  Signature: {sig_name}")
        print(f"    Inputs: {sig_def['inputs']}")
        print(f"    Outputs: {sig_def['outputs']}")


if __name__ == "__main__":
    tflite_path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main/gemma3_1b_main.tflite"
    if os.path.exists(tflite_path):
        print_signatures(tflite_path)
    else:
        print("Model not found at path.")
