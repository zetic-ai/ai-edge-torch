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


def parse_metadata(metadata):
    entries_len = metadata.EntriesLength()
    print(f"System Metadata ({entries_len} entries):")
    for i in range(entries_len):
        entry = metadata.Entries(i)
        key = entry.Key().decode("utf-8")
        vtype = entry.ValueType()
        value = get_vdata_value(entry.Value(), vtype)
        print(f"  - {key}: {value}")


def parse_sections(sections):
    objs_len = sections.ObjectsLength()
    print(f"\nSections ({objs_len} objects):")
    for i in range(objs_len):
        obj = sections.Objects(i)
        begin = obj.BeginOffset()
        end = obj.EndOffset()
        dtype = obj.DataType()
        dtype_str = litertlm_core.any_section_data_type_to_string(dtype)
        print(f"  [{i}] Type: {dtype_str}")
        print(f"      Offset: {begin} - {end} (Size: {end - begin} bytes)")

        items_len = obj.ItemsLength()
        if items_len > 0:
            print(f"      Metadata:")
            for j in range(items_len):
                item = obj.Items(j)
                key = item.Key().decode("utf-8")
                vtype = item.ValueType()
                value = get_vdata_value(item.Value(), vtype)
                print(f"        - {key}: {value}")


def read_litertlm_header(file_path):
    with open(file_path, "rb") as f:
        magic = f.read(8)
        if magic != b"LITERTLM":
            print("Not a LITERTLM file")
            return

        major, minor, patch = struct.unpack("<III", f.read(12))
        print(f"LiteRT-LM Format Version: {major}.{minor}.{patch}")

        f.seek(24)
        header_end_offset = struct.unpack("<Q", f.read(8))[0]

        f.seek(32)
        header_data = f.read(header_end_offset - 32)

        root = schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(header_data, 0)

        sys_metadata = root.SystemMetadata()
        if sys_metadata:
            parse_metadata(sys_metadata)

        sec_metadata = root.SectionMetadata()
        if sec_metadata:
            parse_sections(sec_metadata)


if __name__ == "__main__":
    import sys

    path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/output/gemma-3-270m_dynamic_int8_SM8750_split.litertlm"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    read_litertlm_header(path)
