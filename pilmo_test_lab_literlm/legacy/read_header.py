import struct


def read_litertlm_header(file_path):
    with open(file_path, "rb") as f:
        magic = f.read(8)
        if magic != b"LITERTLM":
            print("Not a LITERTLM file")
            return

        major, minor, patch = struct.unpack("<III", f.read(12))
        print(f"Version: {major}.{minor}.{patch}")

        f.seek(24)
        header_end_offset = struct.unpack("<Q", f.read(8))[0]
        print(f"Header End Offset: {header_end_offset}")

        f.seek(32)
        header_data = f.read(header_end_offset - 32)
        print(f"Header Data Size: {len(header_data)} bytes")
        # Header data is a Flatbuffer.
        # Without the schema, we can look for strings.
        import re

        strings = re.findall(b"[\x20-\x7e]{4,}", header_data)
        print("Strings found in header flatbuffer:")
        for s in strings:
            print(f"  - {s.decode('ascii', errors='ignore')}")


if __name__ == "__main__":
    read_litertlm_header(
        "/home/pilmo/workspace/ai-edge-torch/output/gemma-3-270m_q8_ekv2048.litertlm"
    )
