import struct

import tensorflow.lite.python.schema_py_generated as schema


def rename_tensors_to_match_signatures(tflite_bytes):
    """
    TFLite 바이너리를 분석하여 Signature Names를 실제 Tensor Names로 복사해줍니다.
    AOT 컴파일러가 정상적인 이름을 기반으로 최적화하게 돕습니다.
    """
    buf = bytearray(tflite_bytes)
    model = schema.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)

    # 텐서 인덱스 -> 원하는 시그니처 이름 매핑 수집
    tensor_idx_to_target_name = {}

    for i in range(model.SignatureDefsLength()):
        sig = model.SignatureDefs(i)
        # Outputs
        for j in range(sig.OutputsLength()):
            out = sig.Outputs(j)
            tensor_idx_to_target_name[out.TensorIndex()] = out.Name().decode()
        # Inputs
        for j in range(sig.InputsLength()):
            inp = sig.Inputs(j)
            tensor_idx_to_target_name[inp.TensorIndex()] = inp.Name().decode()

    # 텐서 섹션 순회하며 이름 교체
    # 주의: TFLite 바이너리 내의 String 데이터를 직접 수정하는 것은
    # 기존 코드에서 사용한 surgical_fix와 유사하지만, AOT 전 단계에서 텐서 메타데이터만 건드리는 것이라 훨씬 안전함.

    print("🛠️ Pre-processing TFLite Tensor Names for AOT...")
    for t_idx, target_name in tensor_idx_to_target_name.items():
        tensor = subgraph.Tensors(t_idx)
        old_name = tensor.Name().decode()
        if old_name == target_name:
            continue

        # 텐서의 이름을 target_name으로 강제 패치
        # (문자열 길이가 다르면 바이너리 정렬이 깨지므로,
        # 같은 길이 혹은 짧은 경우만 패딩하여 안전하게 교체하거나
        # 이름을 위해 충분한 버퍼 공간을 확보하는 트릭이 필요함)

        # 여기서는 단순화를 위해 mangled name 패턴을 인식해서 시그니처 이름으로 교환함.
        print(f"  Tensor[{t_idx}]: {old_name} -> {target_name}")

        # 실제 패치 로직 (가장 안전한 방식: 원본 이름을 찾아서 그 자리에 덮어쓰기)
        m_bytes = old_name.encode("utf-8")
        c_bytes = target_name.encode("utf-8")

        # TFLite String은 [4-byte length][data] 구조
        pattern = struct.pack("<I", len(m_bytes)) + m_bytes
        idx = buf.find(pattern)
        if idx != -1:
            # 새 이름이 더 길면 문제지만, StatefulPartitionedCall...은 보통 아주 깁니다.
            if len(c_bytes) <= len(m_bytes):
                buf[idx : idx + 4] = struct.pack("<I", len(c_bytes))
                padded = c_bytes + b"\x00" * (len(m_bytes) - len(c_bytes))
                buf[idx + 4 : idx + 4 + len(m_bytes)] = padded
            else:
                # TODO: 이름이 더 긴 경우 처리 (메모리 재배치 필요)
                pass

    return bytes(buf)
