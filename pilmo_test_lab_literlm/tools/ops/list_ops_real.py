import tensorflow as tf
import os


def list_first_ops(path, limit=100):
    interpreter = tf.lite.Interpreter(model_path=path)
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()

    print(f"--- First {limit} ops of {os.path.basename(path)} ---")
    for i, op in enumerate(ops):
        if i >= limit:
            break
        # Map input indices to names
        in_names = []
        for inp in op["inputs"]:
            if inp < len(tensors):
                in_names.append(f"{tensors[inp]['name']}({inp})")
            else:
                in_names.append(f"UNK({inp})")

        print(f"[{i:2d}] OP: {op['op_name']:20s} In:{in_names} Out:{op['outputs']}")


if __name__ == "__main__":
    list_first_ops(
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model/real_aux.tflite"
    )
