import tensorflow as tf
import os

files = [
    "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8_aot.tflite",
    "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
    "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite",
]

for f in files:
    print(f"\n{'=' * 20} {f} {'=' * 20}")
    if not os.path.exists(f):
        print("MISSING")
        continue
    interpreter = tf.lite.Interpreter(model_path=f)
    sigs = interpreter.get_signature_list()
    for s_name, s_def in sigs.items():
        print(f"\n[Signature: {s_name}]")
        print(f"  Inputs:  {s_def['inputs']}")
        print(f"  Outputs: {s_def['outputs']}")
