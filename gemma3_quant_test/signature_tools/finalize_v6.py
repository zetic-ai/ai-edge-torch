import re
import struct
import sys

try:
    import tensorflow.lite.python.schema_py_generated as schema
except ImportError:
    sys.path.append(
        "/home/pilmo/miniconda3/envs/litertlm/lib/python3.11/site-packages/tensorflow/lite/python"
    )
    import schema_py_generated as schema


def finalize_goldbank_v6(path, out_path):
    print(f"💎 Refurbishing V6 GOLDBANK model: {path}")
    with open(path, "rb") as f:
        buf = bytearray(f.read())

    model = schema.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # 1. Map all output names to official names based on the numeric suffix N
    # decode: StatefulPartitionedCall:N_quantized
    # prefill: StatefulPartitionedCall_1:N_quantized

    patches = {}  # mangled -> official

    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        for j in range(sig.OutputsLength()):
            out = sig.Outputs(j)
            mangled = out.Name().decode()

            if mangled == "logits":
                continue

            # Extract N from :N_
            m = re.search(r":(\d+)_", mangled)
            if m:
                N = int(m.group(1))
                if N < 26:
                    official = f"kv_slice_k_{N}"
                else:
                    official = f"kv_slice_v_{N - 26}"
                patches[mangled] = official

    print(f"  ✅ Identified {len(patches)} names to restore.")

    # 2. Apply string patches safely
    # (Since official names are always shorter than StatefulPartitionedCall..., we use padding)
    for mangled, official in patches.items():
        m_bytes = mangled.encode("utf-8")
        c_bytes = official.encode("utf-8")

        pattern = struct.pack("<I", len(m_bytes)) + m_bytes
        idx = buf.find(pattern)
        if idx != -1:
            buf[idx : idx + 4] = struct.pack("<I", len(c_bytes))
            padded = c_bytes + b"\x00" * (len(m_bytes) - len(c_bytes))
            buf[idx + 4 : idx + 4 + len(m_bytes)] = padded

    with open(out_path, "wb") as f:
        f.write(buf)
    print(f"🎉 FINAL GOLD STANDARD V6 READY: {out_path}")


if __name__ == "__main__":
    finalize_goldbank_v6(
        "gemma3_quant_test/output/optimized_main_v6_w4a16_aot.tflite",
        "gemma3_quant_test/output/optimized_main_v6_w4a16_aot_final.tflite",
    )
