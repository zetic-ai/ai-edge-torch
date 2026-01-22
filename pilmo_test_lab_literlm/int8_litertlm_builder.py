import os
from ai_edge_litert.internal import litertlm_builder
from ai_edge_litert.internal import llm_metadata_pb2
from ai_edge_litert.internal import llm_model_type_pb2


def build_gemma3_int8_litertlm():
    bin_dir = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin"
    output_path = os.path.join(bin_dir, "gemma3_1b_int8_SM8750.litertlm")
    # INT8 모델 경로 설정
    embedder_path = os.path.join(
        bin_dir, "embedder_int8/gemma3_1b_embedder_int8.tflite"
    )
    aux_path = os.path.join(bin_dir, "aux_int8/gemma3_1b_aux_int8.tflite")
    main_path = os.path.join(bin_dir, "main_int8/gemma3_1b_main_int8_aot.tflite")
    tokenizer_path = "/home/pilmo/workspace/ai-edge-torch/LiteRT-LM/runtime/components/testdata/gemma3_sentencepiece.model"
    builder = litertlm_builder.LitertLmFileBuilder()

    # 메타데이터 설정 (Gemma3 1B)
    llm_metadata = llm_metadata_pb2.LlmMetadata()
    llm_metadata.max_num_tokens = 1280
    llm_metadata.start_token.token_ids.ids.append(2)
    for tid in [1, 106, 107]:
        llm_metadata.stop_tokens.add().token_ids.ids.append(tid)
    llm_metadata.llm_model_type.gemma3.CopyFrom(llm_model_type_pb2.Gemma3())
    metadata_pb_path = os.path.join(bin_dir, "gemma3_int8_metadata.pb")
    with open(metadata_pb_path, "wb") as f:
        f.write(llm_metadata.SerializeToString())
    builder.add_llm_metadata(metadata_pb_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)

    # 시스템 메타데이터 추가 (필수)
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="Pilmo",
            dtype=litertlm_builder.DType.STRING,
        )
    )

    # INT8 모델들 추가
    builder.add_tflite_model(embedder_path, litertlm_builder.TfLiteModelType.EMBEDDER)
    # Aux는 보통 AOT된 버전을 사용합니다.
    builder.add_tflite_model(aux_path, litertlm_builder.TfLiteModelType.AUX)
    builder.add_tflite_model(main_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
    with open(output_path, "wb") as f:
        builder.build(f)
    print(f"SUCCESS: {output_path} (INT8) created.")


if __name__ == "__main__":
    build_gemma3_int8_litertlm()
