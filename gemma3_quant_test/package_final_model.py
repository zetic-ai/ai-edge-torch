import os
from ai_edge_litert.internal import litertlm_builder


def main():
    output_dir = "gemma3_quant_test/output"
    final_litertlm_path = os.path.join(output_dir, "gemma3_1b_w4a16_final.litertlm")

    main_model = os.path.join(output_dir, "gemma3_1b_main_w4a16_embed_calib_aot.tflite")
    embedder_model = os.path.join(output_dir, "gemma3_1b_embedder_int8.tflite")
    aux_model = os.path.join(output_dir, "auxiliary.tflite")
    tokenizer_path = os.path.join(output_dir, "tokenizer.model")
    metadata_path = os.path.join(output_dir, "llm_metadata.pb")

    print(f"Building final .litertlm file: {final_litertlm_path}")
    builder = litertlm_builder.LitertLmFileBuilder()

    # 0. Add System Metadata
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors",
            value="Zetic.ai",
            dtype=litertlm_builder.DType.STRING,
        )
    )

    # 1. Add Metadata
    builder.add_llm_metadata(metadata_path)

    # 2. Add Tokenizer
    builder.add_sentencepiece_tokenizer(tokenizer_path)

    # 3. Add Models
    print("Adding Main model (PREFILL_DECODE)...")
    builder.add_tflite_model(
        main_model, litertlm_builder.TfLiteModelType.PREFILL_DECODE
    )

    print("Adding Embedder model...")
    builder.add_tflite_model(embedder_model, litertlm_builder.TfLiteModelType.EMBEDDER)

    print("Adding Auxiliary model...")
    builder.add_tflite_model(aux_model, litertlm_builder.TfLiteModelType.AUX)

    # 4. Build
    with open(final_litertlm_path, "wb") as f:
        builder.build(f)

    print(f"✅ Success! Created final model: {final_litertlm_path}")
    print(f"Size: {os.path.getsize(final_litertlm_path) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
