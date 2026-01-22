import os
import numpy as np
from ai_edge_litert import schema_py_generated as schema
import flatbuffers


def manual_quantize_tflite_tensor(
    input_path, output_path, target_tensor_index, num_bits=8
):
    print(f"Reading model from {input_path}...")
    with open(input_path, "rb") as f:
        model_content = bytearray(f.read())

    model_obj = schema.Model.GetRootAsModel(model_content, 0)
    subgraph = model_obj.Subgraphs(0)

    # 1. Identify target tensor
    tensor = subgraph.Tensors(target_tensor_index)
    tensor_name = tensor.Name().decode("utf-8")
    buffer_idx = tensor.Buffer()
    print(
        f"Targeting Tensor {target_tensor_index}: {tensor_name}, Buffer Index: {buffer_idx}"
    )

    # 2. Extract and Quantize data
    buffer_data = model_obj.Buffers(buffer_idx).DataAsNumpy()
    if buffer_data.size == 0:
        print("ERROR: Buffer is empty. Is this a constant tensor?")
        return

    # Assuming float32
    weights_fp32 = np.frombuffer(buffer_data, dtype=np.float32)
    print(f"Quantizing {weights_fp32.size} elements...")

    v_max = np.max(np.abs(weights_fp32))
    if v_max == 0:
        v_max = 1.0
    scale = v_max / 127.0
    zero_point = 0

    weights_int8 = np.clip(np.round(weights_fp32 / scale), -127, 127).astype(np.int8)

    # 3. Rebuild the model with the new buffer and tensor metadata
    # We use the official TFLite Python API to manipulate if possible, but
    # flatbuffers for manual replacement of buffers is easier for huge weights.

    # Actually, the internal structure of buffers in TFLite is a list of buffers.
    # Buffer 0 is always empty.
    # We need to find the offset of this buffer in the file and replace it.
    # BUT! INT8 is smaller than FP32. If we just overwrite, we leave junk or corrupt the alignment.

    # BETTER: Use a script that rebuilds the flatbuffer properly.

    # I will use a simplified approach:
    # Create a dummy model with the same structure but quantized weights using a temporary script.

    # Actually, I'll use the 'ai_edge_quantizer' again but with a TRICK:
    # I will change the opcode of EMBEDDING_LOOKUP to FULLY_CONNECTED temporarily,
    # quantize it (which will work), then change it back!

    print("Self-Correction: Opcode swapping is too risky.")
    print("I will use the 'tflite' library to create a new model if possible.")


if __name__ == "__main__":
    # We'll use the swap approach in a separate script or just fix the quantizer.
    pass
