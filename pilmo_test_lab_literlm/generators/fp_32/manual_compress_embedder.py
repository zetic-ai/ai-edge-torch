import numpy as np
import os
from ai_edge_litert import schema_py_generated as schema
import flatbuffers


def manual_quantize_embedding(input_path, output_path):
    print(f"Loading model from {input_path}...")
    with open(input_path, "rb") as f:
        model_content = bytearray(f.read())

    model = schema.Model.GetRootAsModel(model_content, 0)
    subgraph = model.Subgraphs(0)

    # 1. Find the embedding weight tensor
    # We know from analysis it's Index 3, but let's be robust
    weight_tensor_idx = -1
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode("utf-8")
        # Look for the huge weight tensor
        shape = t.ShapeAsNumpy()
        if len(shape) == 2 and shape[0] == 262144 and shape[1] == 1152:
            print(f"Found embedding weight tensor: {name} at index {i}")
            weight_tensor_idx = i
            break

    if weight_tensor_idx == -1:
        print("ERROR: Could not find embedding weight tensor!")
        return

    # 2. Extract and Quantize weights
    buffer_idx = subgraph.Tensors(weight_tensor_idx).Buffer()
    buffer = model.Buffers(buffer_idx).DataAsNumpy()

    # Weight tensor is usually float32
    weights_fp32 = np.frombuffer(buffer, dtype=np.float32).reshape(262144, 1152)

    print("Quantizing weights to INT8...")
    # Per-tensor symmetric quantization
    v_max = np.max(np.abs(weights_fp32))
    scale = v_max / 127.0
    weights_int8 = np.clip(np.round(weights_fp32 / scale), -127, 127).astype(np.int8)

    # 3. Create a new model with modified tensor and buffer
    # We use flatbuffers to rebuild if possible, but it's easier to modify in-place
    # if we only change the contents and types.
    # However, changing the buffer size of a flatbuffer in-place is tricky.

    # Use a fresh Builder to create a new model
    builder = flatbuffers.Builder(
        1024 * 1024
    )  # Big enough for metadata, but weights will be added as bytes

    # Since Gemma 1B weights are huge, we should probably just modify the bytearray
    # if we can find the offsets, OR use the schema to rebuild.
    # Rebuilding a 1.2GB model using Python Flatbuffers is EXTREMELY SLOW.

    # BETTER APPROACH: Use the Quantizer to do "Weight Only" on a smaller model
    # to see why it fails, OR use a more efficient tool.

    # Actually, let's try one more trick with ai_edge_quantizer.
    # Maybe EMBEDDING_LOOKUP weights are only quantized if the op is GATHER?
    # No, let's look at the "explicit_dequantize=True" possibility again.

    print(
        "Self-Correction: Rebuilding a 1.2GB flatbuffer in Python is not practical here."
    )
    print(
        "I will try to use ai_edge_quantizer with 'explicit_dequantize=True' specifically."
    )


if __name__ == "__main__":
    # This was a placeholder for analysis.
    # I'll try the explicit_dequantize=True approach in the main script instead.
    pass
