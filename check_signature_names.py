import tensorflow as tf
import os


def analyze_model_io(path):
    print(f"\n{'=' * 20} Analyzing: {path} {'=' * 20}")
    if not os.path.exists(path):
        print("File not found.")
        return

    interpreter = tf.lite.Interpreter(model_path=path)
    sigs = interpreter.get_signature_list()

    # We want to know exactly what the TFLite file reports as signature I/O names.
    # Note: Using private method to get indices if needed, but get_signature_list
    # should show the reported keys.

    for sig_name, sig_def in sigs.items():
        print(f"\n[Signature: {sig_name}]")
        print(f"  Inputs:  {sig_def['inputs']}")
        print(f"  Outputs: {sig_def['outputs']}")


if __name__ == "__main__":
    files = [
        "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8_aot.tflite",  # Use AOT one to be sure
        "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
        "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite",
    ]
    for f in files:
        analyze_model_io(f)
