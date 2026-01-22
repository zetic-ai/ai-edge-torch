import tensorflow as tf


def check_op_details(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    ops = interpreter._get_ops_details()
    op_counts = {}

    print(f"Model: {path}")
    for op in ops:
        op_name = op["op_name"]
        op_counts[op_name] = op_counts.get(op_name, 0) + 1

        if op_name == "FULLY_CONNECTED":
            # Check inputs of FC
            inputs = op["inputs"]
            # FC usually: [input, weight, bias]
            if len(inputs) >= 2:
                weight_idx = inputs[1]
                weight_tensor = interpreter._get_tensor_details(weight_idx)
                print(f"  FC Weight Dtype: {weight_tensor['dtype']}")

    print("\nOp Summary:")
    for name, count in op_counts.items():
        print(f"  - {name:20}: {count}")


check_op_details("pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite")
