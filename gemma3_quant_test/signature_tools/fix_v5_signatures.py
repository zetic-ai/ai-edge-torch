import re
import struct
import sys

# TFLite Schema helper
sys.path.append(
    "/home/pilmo/miniconda3/envs/litertlm/lib/python3.11/site-packages/tensorflow/lite/python"
)
try:
    import schema_py_generated as schema
except ImportError:
    import tensorflow.lite.python.schema_py_generated as schema


def surgical_fix(input_path, output_path):
    print(f"🏥 Starting Surgical Fix (v4.1) on: {input_path}")

    with open(input_path, "rb") as f:
        buf = bytearray(f.read())

    model = schema.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # --- 1. Map Names and Identify Logits position in prefill ---
    patches = {}  # mangled_name -> correct_name
    logits_output_idx = -1

    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        sig_key = sig.SignatureKey().decode("utf-8")

        for j in range(sig.OutputsLength()):
            ot = sig.Outputs(j)
            mangled_name = ot.Name().decode("utf-8")
            t_idx = ot.TensorIndex()
            t = subgraph.Tensors(t_idx)
            shape = [t.Shape(k) for k in range(t.ShapeLength())]

            if shape == [1, 1, 262144]:
                correct_name = "logits"
                if sig_key == "prefill_128":
                    logits_output_idx = j
            else:
                m = re.search(r":(\d+)_", mangled_name)
                if m:
                    N = int(m.group(1))
                    if N < 26:
                        correct_name = f"kv_slice_k_{N}"
                    else:
                        correct_name = f"kv_slice_v_{N - 26}"
                else:
                    correct_name = mangled_name

            patches[mangled_name] = correct_name

    # --- 2. Patch String Data (Names) ---
    print("📝 Patching string names...")
    for mangled, correct in patches.items():
        if mangled == correct:
            continue
        m_bytes = mangled.encode("utf-8")
        c_bytes = correct.encode("utf-8")
        pattern = struct.pack("<I", len(m_bytes)) + m_bytes
        idx = buf.find(pattern)
        if idx != -1:
            buf[idx : idx + 4] = struct.pack("<I", len(c_bytes))
            padded = c_bytes + b"\x00" * (len(m_bytes) - len(c_bytes))
            buf[idx + 4 : idx + 4 + len(m_bytes)] = padded

    # --- 3. Signature Patching: Swap and Truncate ---
    print("✂️ Adjusting prefill_128 structure...")
    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        sig_key = sig.SignatureKey().decode("utf-8")
        if sig_key == "prefill_128":
            # Field 6 is 'outputs' vector in SignatureDef
            off = sig._tab.Offset(6)
            if off:
                vector_ptr = sig._tab.Vector(off)
                length_ptr = vector_ptr - 4
                current_len = struct.unpack("<I", buf[length_ptr : length_ptr + 4])[0]

                print(
                    f"  Current prefill outputs: {current_len}, Logits index: {logits_output_idx}"
                )

                if current_len == 53 and logits_output_idx != -1:
                    # A. Swap Logits element with the last element (index 52)
                    # Each element in the vector is a 4-byte offset
                    logits_ptr = vector_ptr + (logits_output_idx * 4)
                    last_ptr = vector_ptr + (52 * 4)

                    logits_val = buf[logits_ptr : logits_ptr + 4]
                    last_val = buf[last_ptr : last_ptr + 4]

                    buf[logits_ptr : logits_ptr + 4] = last_val
                    buf[last_ptr : last_ptr + 4] = logits_val
                    print(
                        f"  ✅ Swapped index {logits_output_idx}(logits) with index 52"
                    )

                    # B. Truncate the length to 52
                    buf[length_ptr : length_ptr + 4] = struct.pack("<I", 52)
                    print("  ✅ Truncated length to 52")
                else:
                    print(
                        f"  ⚠️ Skipping truncation: length={current_len}, logits_idx={logits_output_idx}"
                    )

    with open(output_path, "wb") as f:
        f.write(buf)
    print(f"🎉 Fixed model saved to: {output_path}")


if __name__ == "__main__":
    surgical_fix(
        "gemma3_quant_test/output/optimized_main_v5_w4a16_aot.tflite",
        "gemma3_quant_test/output/optimized_main_v5_w4a16_aot_final.tflite",
    )
