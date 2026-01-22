import os
import numpy as np
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping


def mimic_quantize(float_tflite_path, output_tflite_path, scale_map):
    print(f"Loading float model: {float_tflite_path}")
    qt = quantizer.Quantizer(float_model=float_tflite_path)

    # 1. Setup Static Recipe
    print("Setting up INT16/INT8 Static Quantization recipe...")
    qt.add_static_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=16,
        weight_num_bits=8,
    )

    # 2. Robust Calibration Bootstrapping
    print("Bootstrapping calibration data with standalone Calibrator...")
    calibration_data = {
        "serving_default": [{"args_0": np.ones((1, 1), dtype=np.int32)}]
    }
    # This fills the internal QSV state
    model_qsvs = qt.calibrate(calibration_data)

    print("Overriding target tensors with official Google scales...")
    # Map from tensor names to (min, max)
    google_overrides = {
        "serving_default_args_0:0": {"min": -32767.0, "max": 32767.0},
    }

    for tensor_name, qsv in google_overrides.items():
        if tensor_name in model_qsvs:
            old_qsv = model_qsvs[tensor_name]
            print(
                f"  [OVERRIDE] {tensor_name}: Scale {old_qsv.get('max', np.array([1.0])).item():.10f} -> [{qsv['min']:.4f}, {qsv['max']:.4f}]"
            )
            model_qsvs[tensor_name] = {
                "min": np.array([qsv["min"]], dtype=np.float32),
                "max": np.array([qsv["max"]], dtype=np.float32),
            }

    # 3. Final Quantization
    print("Quantizing with Hybrid Mimic Calibration data...")
    try:
        result = qt.quantize(calibration_result=model_qsvs)
        result.export_model(output_tflite_path, overwrite=True)
        print(f"SUCCESS: Mimic model saved at {output_tflite_path}")
    except Exception as e:
        print(f"Quantization Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    float_path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/gemma3_gather_embedder.tflite"
    output_path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/gemma3_gather_embedder_mimic.tflite"

    if os.path.exists(float_path):
        mimic_quantize(float_path, output_path, {})
    else:
        print(f"ERROR: Float model not found at {float_path}")
