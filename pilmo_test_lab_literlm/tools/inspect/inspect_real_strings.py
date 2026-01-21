import os


def read_signatures_manually(file_path):
    print(f"\n--- Manual String Inspection: {file_path} ---")
    with open(file_path, "rb") as f:
        data = f.read()

    # 모델 내부에서 텐서 이름으로 쓰일만한 문자열들을 heuristic하게 출력
    # 보통 입력/출력 이름에 'prefill', 'decode', 'tokens', 'logits' 등이 들어감
    keywords = [
        b"prefill",
        b"decode",
        b"logits",
        b"input",
        b"output",
        b"tokens",
        b"pos",
        b"kv_cache",
    ]

    found_strings = set()
    for kw in keywords:
        idx = 0
        while True:
            idx = data.find(kw, idx)
            if idx == -1:
                break

            # 앞뒤로 null 문자가 나올 때까지 읽어서 문자열 추출
            start = idx
            while start > 0 and data[start - 1] >= 32 and data[start - 1] <= 126:
                start -= 1
            end = idx
            while end < len(data) and data[end] >= 32 and data[end] <= 126:
                end += 1

            found_strings.add(data[start:end].decode("utf-8", errors="ignore"))
            idx = end

    for s in sorted(list(found_strings)):
        if len(s) > 2:
            print(f"  Found potential name: {s}")


if __name__ == "__main__":
    base_path = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model"
    )
    read_signatures_manually(os.path.join(base_path, "real_prefill_decode.tflite"))
    read_signatures_manually(os.path.join(base_path, "real_aux.tflite"))
    read_signatures_manually(os.path.join(base_path, "real_embedding.tflite"))
