import tensorflow as tf

files = [
    "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8_aot.tflite",
    "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
    "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite",
]
for f in files:
    print(f"\n--- {f} ---")
    try:
        i = tf.lite.Interpreter(model_path=f)
        sigs = i.get_signature_list()
        for s, d in sigs.items():
            print(f"  Sig {s:25s}: In {d.get('inputs')}, Out {d.get('outputs')}")
    except:
        print(f"  Error loading {f}")
