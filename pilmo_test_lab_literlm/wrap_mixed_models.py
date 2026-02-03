import os

from ai_edge_litert.internal import (
    litertlm_builder,
    llm_metadata_pb2,
    llm_model_type_pb2,
)


def build_mixed_litertlm():
    base_dir = "/home/pilmo/workspace/ai-edge-torch"
    ref_dir = os.path.join(
        base_dir, "pilmo_test_lab_literlm", "reference", "succeed_model"
    )
    our_dir = os.path.join(base_dir, "gemma3_quant_test", "output")
    output_dir = os.path.join(base_dir, "pilmo_test_lab_literlm", "bin", "mixed_test")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "PILMO_LOVE_GEMMA.litertlm")

    # 1. Component Paths
    # Reference Tokenizer
    tokenizer_path = "/home/pilmo/workspace/ai-edge-torch/LiteRT-LM/runtime/components/testdata/gemma3_sentencepiece.model"

    # Reference Embedder and Aux
    embedder_path = os.path.join(ref_dir, "real_embedding.tflite")
    aux_path = os.path.join(ref_dir, "real_aux.tflite")

    # OUR Optimized Main Model
    main_path = os.path.join(
        our_dir,
        "pilmo_optimized_main_w4a16_aot_FINAL.tflite",
    )

    # Verify paths
    for p in [tokenizer_path, embedder_path, aux_path, main_path]:
        if not os.path.exists(p):
            print(f"ERROR: Path {p} does not exist.")
            # If our main model is missing, we might need to run the export script.
            if p == main_path:
                print("Hint: Run gemma3_quant_test/final_optimized_export_v3.py first.")
            return

    # 2. Create LLM Metadata (Same as reference spec)
    print("Generating LLM Metadata...")
    llm_metadata = llm_metadata_pb2.LlmMetadata()
    llm_metadata.max_num_tokens = 1280
    llm_metadata.start_token.token_ids.ids.append(2)
    for tid in [1, 106, 107]:
        stop_tok = llm_metadata.stop_tokens.add()
        stop_tok.token_ids.ids.append(tid)
    llm_metadata.llm_model_type.gemma3.CopyFrom(llm_model_type_pb2.Gemma3())

    metadata_pb_path = os.path.join(output_dir, "mixed_metadata.pb")
    with open(metadata_pb_path, "wb") as f:
        f.write(llm_metadata.SerializeToString())

    # 3. Initialize the Builder
    builder = litertlm_builder.LitertLmFileBuilder()

    # 4. Add System Metadata
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="Zetic.ai (Mixed Verification)",
            dtype=litertlm_builder.DType.STRING,
        )
    )

    # 5. Add LLM Metadata and Tokenizer
    print(f"Adding LLM Metadata: {metadata_pb_path}")
    builder.add_llm_metadata(metadata_pb_path)

    print(f"Adding Tokenizer: {tokenizer_path}")
    builder.add_sentencepiece_tokenizer(tokenizer_path)

    # 6. Add Models
    print("Adding Models...")
    print(f"  [Embedder (REF)]: {embedder_path}")
    builder.add_tflite_model(embedder_path, litertlm_builder.TfLiteModelType.EMBEDDER)

    print(f"  [Aux (REF)]: {aux_path}")
    builder.add_tflite_model(aux_path, litertlm_builder.TfLiteModelType.AUX)

    print(f"  [Main (OURS)]: {main_path}")
    builder.add_tflite_model(main_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE)

    # 7. Final Build
    print(f"Building mixed .litertlm at {output_path}...")
    try:
        with open(output_path, "wb") as f:
            builder.build(f)
        print("\n" + "=" * 50)
        print("SUCCESS! Mixed LiteRT-LM Package is ready.")
        print(f"Output: {output_path}")
        print("=" * 50)
    except Exception as e:
        print(f"BUILD ERROR: {e}")


if __name__ == "__main__":
    build_mixed_litertlm()
