import tensorflow as tf
import os


def inspect_tflite(file_path):
    print(f"--- Inspecting {file_path} ---")
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()

    # 텐서 정보 출력
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    print("\n[Input Details]")
    for i, detail in enumerate(inputs):
        print(
            f"Input {i}: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}"
        )

    print("\n[Output Details]")
    for i, detail in enumerate(outputs):
        print(
            f"Output {i}: {detail['name']}, Shape: {detail['shape']}, Dtype: {detail['dtype']}"
        )

    # 오퍼레이터 수 확인 (이건 interpreter에서 바로는 어렵고 모델 데이터를 직접 파싱해야 함)
    # 하지만 시그니처 이름만 봐도 힌트를 얻을 수 있음


if __name__ == "__main__":
    base_path = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model"
    )
    inspect_tflite(os.path.join(base_path, "real_prefill_decode.tflite"))
    inspect_tflite(os.path.join(base_path, "real_aux.tflite"))
    inspect_tflite(os.path.join(base_path, "real_embedding.tflite"))
