#!/usr/bin/env python3
import argparse
import os
import struct
import flatbuffers
from ai_edge_litert.internal import litertlm_header_schema_py_generated as schema
from ai_edge_litert.internal import litertlm_core


def get_vdata_value(vdata, vdata_type):
    if vdata_type == schema.VData.Bool:
        val = schema.Bool()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Int8:
        val = schema.Int8()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Int16:
        val = schema.Int16()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Int32:
        val = schema.Int32()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Int64:
        val = schema.Int64()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.UInt8:
        val = schema.UInt8()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.UInt16:
        val = schema.UInt16()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.UInt32:
        val = schema.UInt32()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.UInt64:
        val = schema.UInt64()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Float32:
        val = schema.Float32()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.Double:
        val = schema.Double()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value()
    elif vdata_type == schema.VData.StringValue:
        val = schema.StringValue()
        val.Init(vdata.Bytes, vdata.Pos)
        return val.Value().decode("utf-8")
    return None


def get_filename_for_type(dtype, index):
    mapping = {
        schema.AnySectionDataType.LlmMetadataProto: "llm_metadata.pb",
        schema.AnySectionDataType.SP_Tokenizer: "tokenizer.model",
        schema.AnySectionDataType.HF_Tokenizer_Zlib: "tokenizer.json.zlib",
        schema.AnySectionDataType.TFLiteModel: "model.tflite",
    }
    base = mapping.get(dtype, f"section_{index}")
    if dtype not in mapping:
        return f"{base}.bin"
    return base


def unpack_litertlm(input_path, output_dir):
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_path, "rb") as f:
        # 1. Read Magic and Version
        magic = f.read(8)
        if magic != b"LITERTLM":
            print("Error: Not a valid LITERTLM file.")
            return

        major, minor, patch = struct.unpack("<III", f.read(12))
        print(f"[*] LiteRT-LM Package Version: {major}.{minor}.{patch}")

        # 2. Get Header End Offset
        f.seek(24)
        header_end = struct.unpack("<Q", f.read(8))[0]

        # 3. Read and Parse Flatbuffer Header
        f.seek(32)
        header_data = f.read(header_end - 32)
        root = schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(header_data, 0)

        sec_metadata = root.SectionMetadata()
        num_sections = sec_metadata.ObjectsLength()
        print(f"[*] Found {num_sections} sections in package.\n")

        for i in range(num_sections):
            obj = sec_metadata.Objects(i)
            dtype = obj.DataType()
            begin = obj.BeginOffset()
            end = obj.EndOffset()
            size = end - begin

            dtype_str = litertlm_core.any_section_data_type_to_string(dtype)
            filename = get_filename_for_type(dtype, i)
            save_path = os.path.join(output_dir, filename)

            # Check for conflict and add index if needed
            if os.path.exists(save_path):
                save_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}",
                )

            print(f"[Section {i}]")
            print(f"  Type: {dtype_str}")
            print(f"  Range: {begin} - {end} ({size / (1024 * 1024):.2f} MB)")

            # Print Metadata if any
            items_len = obj.ItemsLength()
            for j in range(items_len):
                item = obj.Items(j)
                k = item.Key().decode("utf-8")
                v = get_vdata_value(item.Value(), item.ValueType())
                print(f"  Metadata: {k} = {v}")

            # Extract data
            print(f"  Extracting to -> {save_path} ...", end="", flush=True)
            f.seek(begin)
            with open(save_path, "wb") as out_f:
                chunk_size = 1024 * 1024  # 1MB
                remaining = size
                while remaining > 0:
                    out_f.write(f.read(min(chunk_size, remaining)))
                    remaining -= chunk_size
            print(" Done")
            print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="LiteRT-LM Unpacker CLI tool")
    parser.add_argument("input", help="Path to the .litertlm file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory (default: same as input name)",
        default=None,
    )

    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output

    if output_dir is None:
        output_dir = os.path.splitext(os.path.basename(input_file))[0] + "_unpacked"

    unpack_litertlm(input_file, output_dir)


if __name__ == "__main__":
    main()
