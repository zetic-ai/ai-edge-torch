import tensorflow as tf


def check_op_details(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    tensor_details = interpreter.get_tensor_details()

    print(f"Model: {path}")
    count = 0
    for t in tensor_details:
        if len(t["shape"]) == 2:
            # Check if it's likely a weight (large 2D)
            if t["shape"][0] > 100 or t["shape"][1] > 100:
                print(
                    f"  Tensor: {t['name']:50} | Shape: {str(t['shape']):15} | Dtype: {t['dtype']}"
                )
                count += 1
                if count >= 30:
                    break


check_op_details("pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite")
