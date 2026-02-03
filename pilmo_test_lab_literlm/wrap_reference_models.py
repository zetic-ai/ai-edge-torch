import os

from ai_edge_litert.internal import (
    litertlm_builder,
    llm_metadata_pb2,
    llm_model_type_pb2,
)


def build_reference_litertlm():
    base_dir = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm"
    ref_dir = os.path.join(base_dir, "reference", "succeed_model")
    output_dir = os.path.join(base_dir, "bin", "reference_test")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "gemma3_reference_official.litertlm")

    # 1. Component Paths (Reference Models)
    tokenizer_path = "/home/pilmo/workspace/ai-edge-torch/LiteRT-LM/runtime/components/testdata/gemma3_sentencepiece.model"

    embedder_path = os.path.join(ref_dir, "real_embedding.tflite")
    aux_path = os.path.join(ref_dir, "real_aux.tflite")
    main_path = os.path.join(ref_dir, "real_prefill_decode.tflite")

    # Verify paths
    for p in [tokenizer_path, embedder_path, aux_path, main_path]:
        if not os.path.exists(p):
            print(f"ERROR: Path {p} does not exist.")
            return

    # 2. Create LLM Metadata (Protobuf)
    print("Generating LLM Metadata...")
    llm_metadata = llm_metadata_pb2.LlmMetadata()
    llm_metadata.max_num_tokens = 1280

    # Set Start Token
    llm_metadata.start_token.token_ids.ids.append(2)
    # Set Stop Tokens (EOS=1, and others)
    for tid in [1, 106, 107]:
        stop_tok = llm_metadata.stop_tokens.add()
        stop_tok.token_ids.ids.append(tid)

    # Set Model Type to GEMMA3
    llm_metadata.llm_model_type.gemma3.CopyFrom(llm_model_type_pb2.Gemma3())

    # Save temporary .pb for the builder
    metadata_pb_path = os.path.join(output_dir, "reference_metadata.pb")
    with open(metadata_pb_path, "wb") as f:
        f.write(llm_metadata.SerializeToString())

    # 3. Initialize the Builder
    builder = litertlm_builder.LitertLmFileBuilder()

    # 4. Add System Metadata
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="Google AI Edge (Reference Wrap)",
            dtype=litertlm_builder.DType.STRING,
        )
    )

    # 5. Add LLM Metadata and Tokenizer
    print(f"Adding LLM Metadata: {metadata_pb_path}")
    builder.add_llm_metadata(metadata_pb_path)

    print(f"Adding Tokenizer: {tokenizer_path}")
    builder.add_sentencepiece_tokenizer(tokenizer_path)

    # 6. Add Models
    print("Adding Reference TFLite Models...")
    builder.add_tflite_model(embedder_path, litertlm_builder.TfLiteModelType.EMBEDDER)
    builder.add_tflite_model(aux_path, litertlm_builder.TfLiteModelType.AUX)
    builder.add_tflite_model(main_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE)

    # 7. Final Build
    print(f"Building reference .litertlm at {output_path}...")
    try:
        with open(output_path, "wb") as f:
            builder.build(f)
        print("\n" + "=" * 50)
        print("SUCCESS! Reference LiteRT-LM Package is ready.")
        print(f"Output: {output_path}")
        print("=" * 50)
    except Exception as e:
        print(f"BUILD ERROR: {e}")


if __name__ == "__main__":
    build_reference_litertlm()
