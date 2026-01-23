#!/usr/bin/env python3
"""Analyze TFLite model for dispatch ops and NPU fusion."""

import sys
import tensorflow as tf
from collections import defaultdict


def analyze_tflite_model(model_path):
    """Analyze TFLite model structure and dispatch ops."""

    # Load interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get tensor details
    tensor_details = interpreter.get_tensor_details()

    print("=" * 80)
    print(f"TFLite Model Analysis: {model_path.split('/')[-1]}")
    print("=" * 80)

    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  • Total Inputs:  {len(input_details)}")
    print(f"  • Total Outputs: {len(output_details)}")
    print(f"  • Total Tensors: {len(tensor_details)}")

    # I/O types
    input_types = set(inp["dtype"].__name__ for inp in input_details)
    output_types = set(out["dtype"].__name__ for out in output_details)

    print(f"\n🔢 I/O Data Types:")
    print(f"  • Input Types:  {', '.join(sorted(input_types))}")
    print(f"  • Output Types: {', '.join(sorted(output_types))}")

    # Analyze operators
    print(f"\n🔧 Operator Analysis:")

    # Count operators by type
    op_counts = defaultdict(int)
    dispatch_ops = []

    for tensor in tensor_details:
        # Check if this is a dispatch op (custom op for NPU)
        if "name" in tensor:
            name = tensor["name"]
            if "dispatch" in name.lower() or "DISPATCH" in name:
                dispatch_ops.append(name)

            # Extract op type from tensor name
            if "/" in name:
                parts = name.split("/")
                for part in parts:
                    if any(
                        op_keyword in part
                        for op_keyword in [
                            "MatMul",
                            "Conv",
                            "Add",
                            "Mul",
                            "Concat",
                            "Softmax",
                            "Reshape",
                            "Transpose",
                            "Gather",
                            "Cast",
                        ]
                    ):
                        op_counts[part.split("_")[0]] += 1

    # Print operator counts
    if op_counts:
        print(f"  • Operator Counts:")
        for op_type, count in sorted(
            op_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            print(f"    - {op_type}: {count}")

    # Dispatch ops
    print(f"\n🚀 Dispatch Op Analysis:")
    if dispatch_ops:
        print(f"  • Total Dispatch Ops: {len(dispatch_ops)}")
        print(f"  • Dispatch Op Names:")
        for i, op_name in enumerate(dispatch_ops[:5], 1):
            print(f"    {i}. {op_name}")
        if len(dispatch_ops) > 5:
            print(f"    ... and {len(dispatch_ops) - 5} more")
    else:
        print(f"  • No explicit dispatch ops found in tensor names")
        print(f"  • Note: Dispatch ops may be embedded in custom ops")

    # Check for custom ops (NPU delegation indicator)
    custom_ops = []
    for tensor in tensor_details:
        if "name" in tensor:
            name = tensor["name"]
            if "custom" in name.lower() or "delegate" in name.lower():
                custom_ops.append(name)

    if custom_ops:
        print(f"\n🎯 Custom/Delegate Ops (NPU Indicators):")
        print(f"  • Total Custom Ops: {len(custom_ops)}")
        for i, op_name in enumerate(custom_ops[:3], 1):
            print(f"    {i}. {op_name}")

    # Quantization analysis
    print(f"\n⚖️ Quantization Analysis:")
    quantized_tensors = 0
    quant_types = defaultdict(int)

    for tensor in tensor_details:
        if "quantization_parameters" in tensor:
            quant_params = tensor["quantization_parameters"]
            if (
                quant_params.get("scales") is not None
                and len(quant_params["scales"]) > 0
            ):
                quantized_tensors += 1
                dtype = tensor.get("dtype", "unknown")
                if hasattr(dtype, "__name__"):
                    quant_types[dtype.__name__] += 1

    print(f"  • Quantized Tensors: {quantized_tensors} / {len(tensor_details)}")
    if quant_types:
        print(f"  • Quantization Types:")
        for dtype, count in sorted(
            quant_types.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    - {dtype}: {count}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_dispatch_ops.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    analyze_tflite_model(model_path)
