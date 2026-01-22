import tensorflow as tf
import os


def get_io_details(path, sig_name):
    if not os.path.exists(path):
        return None
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        sigs = interpreter.get_signature_list()
        if sig_name not in sigs:
            return f"Sig {sig_name} not found"

        sig_def = sigs[sig_name]
        tensor_details = {t["index"]: t for t in interpreter.get_tensor_details()}

        inputs = {}
        # Handle both list and dict formats of 'inputs'
        if isinstance(sig_def["inputs"], dict):
            for name, idx in sig_def["inputs"].items():
                t = tensor_details.get(idx)
                inputs[name] = (t["shape"], t["dtype"])
        else:  # list format
            # In list format, name is the tensor name in tensor_details
            # We have to find it by name
            for name in sig_def["inputs"]:
                for t in interpreter.get_tensor_details():
                    if t["name"] == name:
                        inputs[name] = (t["shape"], t["dtype"])
                        break

        outputs = {}
        if isinstance(sig_def["outputs"], dict):
            for name, idx in sig_def["outputs"].items():
                t = tensor_details.get(idx)
                outputs[name] = (t["shape"], t["dtype"])
        else:
            for name in sig_def["outputs"]:
                for t in interpreter.get_tensor_details():
                    if t["name"] == name:
                        outputs[name] = (t["shape"], t["dtype"])
                        break
        return inputs, outputs
    except Exception as e:
        return str(e), None


def analyze_pipeline():
    models = {
        "Embedder (Decode)": (
            "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8_aot.tflite",
            "decode_embedder",
        ),
        "Aux (Rope Decode)": (
            "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
            "decode_rope",
        ),
        "Aux (Mask Decode)": (
            "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
            "decode_mask",
        ),
        "Main (Decode)": (
            "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite",
            "decode",
        ),
        "Aux (Cache Update Decode)": (
            "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8_aot.tflite",
            "decode_cache_update",
        ),
    }

    print("\n" + "=" * 80)
    print(f"{'STEP':40} | {'TENSORS (NAME, SHAPE, DTYPE)'}")
    print("=" * 80)
    for desc, (path, sig) in models.items():
        res = get_io_details(path, sig)
        if isinstance(res, tuple):
            inputs, outputs = res
            print(f"\n[{desc}]")
            print("  Inputs:")
            for n, (s, d) in inputs.items():
                print(f"    - {n:30} : {str(s):15} {d}")
            print("  Outputs:")
            for n, (s, d) in outputs.items():
                print(f"    - {n:30} : {str(s):15} {d}")
        else:
            print(f"\n[{desc}] Error: {res}")


analyze_pipeline()
