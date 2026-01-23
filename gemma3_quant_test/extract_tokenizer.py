import struct
import os
from ai_edge_litert.internal import litertlm_header_schema_py_generated as schema


def extract_tokenizer(src_path, dst_path):
    with open(src_path, "rb") as f:
        f.seek(8)
        # Skip magic
        f.seek(24)
        header_end_offset = struct.unpack("<Q", f.read(8))[0]
        f.seek(32)
        header_data = f.read(header_end_offset - 32)
        root = schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(header_data, 0)
        sec_metadata = root.SectionMetadata()

        for i in range(sec_metadata.ObjectsLength()):
            obj = sec_metadata.Objects(i)
            if obj.DataType() == schema.AnySectionDataType.SP_Tokenizer:
                begin = obj.BeginOffset()
                end = obj.EndOffset()
                size = end - begin
                print(f"Found SP_Tokenizer at {begin}-{end}, size {size}")
                f.seek(begin)
                data = f.read(size)
                with open(dst_path, "wb") as out_f:
                    out_f.write(data)
                print(f"Extracted to {dst_path}")
                return True
    return False


if __name__ == "__main__":
    src = "resources/litertlm_download/Gemma3-1B-IT_q4_ekv1280_sm8750.litertlm"
    dst = "gemma3_quant_test/output/tokenizer.model"
    extract_tokenizer(src, dst)
