import tensorflow as tf
import numpy as np
import os


def analyze_quant(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    try:
        interpreter.allocate_tensors()
    except:
        pass

    tensor_details = interpreter.get_tensor_details()

    print("=" * 90)
    print(f"Detailed Quantization Analysis: {model_path}")
    print("=" * 90)

    # Get I/O indices
    inputs = [t["index"] for t in interpreter.get_input_details()]
    outputs = [t["index"] for t in interpreter.get_output_details()]

    # Collect indices to show
    indices_to_show = inputs + outputs

    # Add some internal int16 tensors
    count = 0
    for t in tensor_details:
        if t["dtype"] == np.int16 and t["index"] not in indices_to_show:
            indices_to_show.append(t["index"])
            count += 1
            if count > 20:
                break

    header = "{:<6} | {:<45} | {:<8} | {:<12} | {:<5}".format(
        "Index", "Name", "Dtype", "Scale", "ZP"
    )
    print(header)
    print("-" * 90)

    for idx in indices_to_show:
        t = tensor_details[idx]
        name = t["name"]
        dtype = t["dtype"].__name__
        quant = t.get("quantization_parameters", {})
        scales = quant.get("scales", [])
        zps = quant.get("zero_points", [])

        scale_val = "{:.2e}".format(scales[0]) if len(scales) > 0 else "None"
        zp_val = str(zps[0]) if len(zps) > 0 else "None"

        print(
            "{:<6} | {:<45} | {:<8} | {:<12} | {:<5}".format(
                idx, name[:45], dtype, scale_val, zp_val
            )
        )


if __name__ == "__main__":
    analyze_quant(
        "pilmo_test_lab_literlm/reference/succeed_model/real_prefill_decode.tflite"
    )
