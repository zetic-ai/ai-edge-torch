import tensorflow as tf
import sys


def check_op_details(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    tensor_details = interpreter.get_tensor_details()

    print(f"Model: {path}")
    count = 0
    for t in tensor_details:
        if len(t["shape"]) == 2:
            if t["shape"][0] > 100 or t["shape"][1] > 100:
                print(
                    f"  Tensor: {t['name']:50} | Shape: {str(t['shape']):15} | Dtype: {t['dtype']}"
                )
                count += 1
                if count >= 30:
                    break


if len(sys.argv) > 1:
    check_op_details(sys.argv[1])
else:
    print("Usage: python check_fc_types_v4.py <model_path>")
