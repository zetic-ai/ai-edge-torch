import tensorflow as tf


def check_op_details(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    # Get all ops manually using the internal API carefully
    # Or just use the summary

    # We can get tensor details for the main subgraph (0)
    tensor_details = {t["index"]: t for t in interpreter.get_tensor_details()}

    # Iterate through ops
    # TFLite interpreter doesn't expose _get_ops_details easily in newer versions
    # We can use the summarize_tflite.py approach but I'll try to find FC weights

    print(f"Model: {path}")
    count = 0
    for t_idx, t in tensor_details.items():
        # Heuristic: weights of FULLY_CONNECTED are often named with 'weight'
        # or are constant tensors (though hard to tell from here without op graph)
        if "weight" in t["name"].lower() and len(t["shape"]) == 2:
            print(f"  Tensor: {t['name']:50} | Dtype: {t['dtype']}")
            count += 1
            if count >= 20:
                break


check_op_details("pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite")
