import struct
import os
from ai_edge_litert.internal import litertlm_header_schema_py_generated as schema
from ai_edge_litert.internal import litertlm_core


def extract_tflite_from_litertlm(src_path, dst_path=None):
    if not os.path.exists(src_path):
        print(f"Source file not found: {src_path}")
        return

    with open(src_path, "rb") as f:
        # 1. 헤더 체크
        magic = f.read(8)
        if magic != b"LITERTLM":
            print("Error: Not a LITERTLM file.")
            return

        # 2. 메타데이터 끝 위치 확인
        f.seek(24)
        header_end_offset = struct.unpack("<Q", f.read(8))[0]

        # 3. Flatbuffer 섹션 맵 읽기
        f.seek(32)
        header_data = f.read(header_end_offset - 32)
        root = schema.LiteRTLMMetaData.GetRootAsLiteRTLMMetaData(header_data, 0)
        sec_metadata = root.SectionMetadata()

        found_count = 0
        for i in range(sec_metadata.ObjectsLength()):
            obj = sec_metadata.Objects(i)
            # TFLiteModel 타입 섹션인 경우 처리
            if obj.DataType() != schema.AnySectionDataType.TFLiteModel:
                continue

            found_count += 1
            tflite_section = obj

            # 메타데이터(Items)에서 model_type 찾기
            model_type = f"model_{found_count}"
            if obj.ItemsLength() > 0:
                for j in range(obj.ItemsLength()):
                    kv = obj.Items(j)
                    if kv.Key().decode() == "model_type":
                        vtype = kv.ValueType()
                        if vtype == schema.VData.StringValue:
                            val = schema.StringValue()
                            val.Init(kv.Value().Bytes, kv.Value().Pos)
                            model_type = val.Value().decode("utf-8")
                        break

            begin = tflite_section.BeginOffset()
            end = tflite_section.EndOffset()
            size = end - begin

            # 출력 경로 설정 (원본파일명_model_type.tflite)
            current_dst_path = os.path.splitext(src_path)[0] + f"_{model_type}.tflite"

            print(f"Extracting TFLite model {found_count}...")
            print(f"  Model Type: {model_type}")
            print(f"  Offset: {begin} - {end}")
            print(f"  Size: {size / (1024 * 1024):.2f} MB")

            # 데이터 추출 및 저장
            original_pos = f.tell()
            f.seek(begin)
            with open(current_dst_path, "wb") as out_f:
                chunk_size = 1024 * 1024 * 10  # 10MB chunks
                remaining = size
                while remaining > 0:
                    to_read = min(chunk_size, remaining)
                    data = f.read(to_read)
                    out_f.write(data)
                    remaining -= len(data)
            f.seek(original_pos)

            print(f"Successfully extracted to: {current_dst_path}\n")

        if found_count == 0:
            print("Error: No TFLite model sections found in this file.")
            return

        print(f"Successfully extracted to: {dst_path}")


if __name__ == "__main__":
    import sys

    litertlm_path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/output/gemma-3-270m_dynamic_int8_SM8750_split_v4_final.litertlm"
    if len(sys.argv) > 1:
        litertlm_path = sys.argv[1]

    extract_tflite_from_litertlm(litertlm_path)
