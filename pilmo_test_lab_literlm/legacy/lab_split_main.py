import os
import torch
import tempfile
import argparse

import ai_edge_torch
from ai_edge_torch.generative.utilities import converter as gen_converter

# Internal imports for bundling
from ai_edge_litert.internal import litertlm_builder as lm_builder
from ai_edge_litert.internal import llm_metadata_pb2
from ai_edge_litert.internal import llm_model_type_pb2

# Local imports
from registry.models import MODEL_SPECS
from core.utils import download_from_hf
from core import split_utils
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


def run_split_lab_v4_mimic(
    model_key, quant, out_dir, prefill_len, kv_len, target_soc="SM8750"
):
    spec = MODEL_SPECS[model_key]
    checkpoint_dir = download_from_hf(spec["repo_id"])

    print(f"\n[1/4] Building PyTorch modules for {model_key} Mimic...")
    full_model = spec["builder"](checkpoint_dir, mask_cache_size=kv_len)
    full_model.eval()

    config = full_model.config
    num_layers = config.num_layers
    head_dim = config.block_config(0).attn_config.head_dim
    head_count = config.block_config(0).attn_config.num_query_groups
    emb_dim = config.embedding_dim

    embedder_mod = split_utils.Gemma3Embedder(full_model)
    rope_mod = split_utils.Gemma3Rope(config)
    mask_mod = split_utils.Gemma3Mask(kv_len=kv_len)

    MainClass = split_utils.get_main_module_class(num_layers)
    main_mod = MainClass(full_model)

    CacheUpdateClass = split_utils.get_cache_update_module_class(num_layers)
    cache_update_mod = CacheUpdateClass(num_layers)

    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:

        def sample_tokens(s):
            return torch.randint(0, 100, (1, s), dtype=torch.int32)

        def sample_floats(shape):
            return torch.ones(shape, dtype=torch.float32)

        def sample_pos(s):
            return torch.arange(s, dtype=torch.int32)

        # --- Step A: Export Embedder ---
        print(f"\n[2/4] Exporting Embedder.tflite...")
        emb_conv = ai_edge_torch._convert.converter.Converter()
        emb_conv.add_signature(
            f"prefill_embedder_{prefill_len}",
            embedder_mod,
            sample_kwargs={"token_ids": sample_tokens(prefill_len)},
        )
        emb_conv.add_signature(
            "decode_embedder",
            embedder_mod,
            sample_kwargs={"token_ids": sample_tokens(1)},
        )
        edge_embedder = emb_conv.convert()
        embedder_tflite_path = os.path.join(workdir, "embedder.tflite")
        edge_embedder.export(embedder_tflite_path)

        # --- Step B: Export Aux ---
        print(f"\n[B] Exporting Aux.tflite...")
        aux_conv = ai_edge_torch._convert.converter.Converter()
        aux_conv.add_signature(
            f"prefill_rope_{prefill_len}",
            rope_mod,
            sample_kwargs={"input_pos": sample_pos(prefill_len)},
        )
        aux_conv.add_signature(
            "decode_rope", rope_mod, sample_kwargs={"input_pos": sample_pos(1)}
        )
        aux_conv.add_signature(
            f"prefill_mask_{prefill_len}",
            mask_mod,
            sample_kwargs={
                "time_step": torch.tensor([0], dtype=torch.int32),
                "input_tokens": sample_tokens(prefill_len),
            },
        )
        aux_conv.add_signature(
            "decode_mask",
            mask_mod,
            sample_kwargs={
                "time_step": torch.tensor([0], dtype=torch.int32),
                "input_tokens": sample_tokens(1),
            },
        )

        def get_cu_kwargs():
            d = {"input_pos": torch.tensor([0], dtype=torch.int32)}
            for i in range(num_layers):
                d[f"kv_cache_k_{i}"] = sample_floats((1, head_count, kv_len, head_dim))
                d[f"kv_cache_v_{i}"] = sample_floats((1, head_count, head_dim, kv_len))
                d[f"kv_slice_k_{i}"] = sample_floats((1, head_count, 1, head_dim))
                d[f"kv_slice_v_{i}"] = sample_floats((1, head_count, head_dim, 1))
            return d

        aux_conv.add_signature(
            f"prefill_cache_update_{prefill_len}",
            cache_update_mod,
            sample_kwargs=get_cu_kwargs(),
        )
        aux_conv.add_signature(
            "decode_cache_update", cache_update_mod, sample_kwargs=get_cu_kwargs()
        )

        edge_aux = aux_conv.convert()
        aux_tflite_path = os.path.join(workdir, "aux.tflite")
        edge_aux.export(aux_tflite_path)

        # --- Step C: Export Main ---
        print(f"\n[3/4] Exporting Main Transformer (NPU Clean AOT)...")
        soc_model = getattr(
            qnn_target.SocModel, target_soc.upper(), qnn_target.SocModel.SM8750
        )
        qnn_backend = qnn_target.Target(soc_model=soc_model)
        ai_edge_torch.experimental_add_compilation_backend(qnn_backend)

        main_conv = ai_edge_torch._convert.converter.Converter()

        def get_main_kwargs(cur_s):
            d = {
                "embeddings": sample_floats((1, cur_s, emb_dim)),
                "input_pos": sample_pos(cur_s),
                "mask_global": sample_floats((1, 1, cur_s, kv_len)),
                "mask_local": sample_floats((1, 1, cur_s, kv_len)),
                "pos_emb_cos": sample_floats((1, 1, cur_s, head_dim // 2)),
                "pos_emb_sin": sample_floats((1, 1, cur_s, head_dim // 2)),
                "pos_emb_local_cos": sample_floats((1, 1, cur_s, head_dim // 2)),
                "pos_emb_local_sin": sample_floats((1, 1, cur_s, head_dim // 2)),
            }
            for i in range(num_layers):
                d[f"kv_cache_k_{i}"] = sample_floats((1, head_count, kv_len, head_dim))
                d[f"kv_cache_v_{i}"] = sample_floats((1, head_count, head_dim, kv_len))
            return d

        main_conv.add_signature(
            f"prefill_{prefill_len}",
            main_mod,
            sample_kwargs=get_main_kwargs(prefill_len),
        )
        main_conv.add_signature("decode", main_mod, sample_kwargs=get_main_kwargs(1))

        quant_config = gen_converter.get_quant_recipe_from_flag(quant, config)
        edge_main = main_conv.convert(quant_config=quant_config, strict_export=False)
        main_tflite_path = os.path.join(workdir, "main.tflite")
        edge_main.export(main_tflite_path)

        # --- Step D: Bundling ---
        print(f"\n[4/4] Final Bundling...")
        final_filename = f"{model_key}_{quant}_{target_soc}_split_v4_final.litertlm"
        final_path = os.path.join(out_dir, final_filename)

        llm_meta = llm_metadata_pb2.LlmMetadata()
        m_spec = spec["litertlm_metadata"]
        llm_meta.max_num_tokens = kv_len
        llm_meta.start_token.token_ids.ids.append(m_spec["start_token_id"])
        for tid in m_spec["stop_token_ids"]:
            llm_meta.stop_tokens.add().token_ids.ids.append(tid)
        llm_meta.llm_model_type.gemma3.CopyFrom(llm_model_type_pb2.Gemma3())
        meta_pb_path = os.path.join(workdir, "metadata.pb")
        with open(meta_pb_path, "wb") as f:
            f.write(llm_meta.SerializeToString())

        builder = lm_builder.LitertLmFileBuilder()
        builder.add_system_metadata(
            lm_builder.Metadata(
                key="Authors",
                value="AI Edge Torch V4 Final",
                dtype=lm_builder.DType.STRING,
            )
        )
        builder.add_llm_metadata(meta_pb_path)
        builder.add_sentencepiece_tokenizer(
            os.path.join(checkpoint_dir, "tokenizer.model")
        )
        builder.add_tflite_model(
            embedder_tflite_path, lm_builder.TfLiteModelType.EMBEDDER
        )
        builder.add_tflite_model(aux_tflite_path, lm_builder.TfLiteModelType.AUX)
        builder.add_tflite_model(
            main_tflite_path, lm_builder.TfLiteModelType.PREFILL_DECODE
        )

        with open(final_path, "wb") as f:
            builder.build(f)
        print(f"SUCCESS: {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemma-3-270m")
    args = parser.parse_args()
    # Let's run 270m now since we verified 1b worked for shapes
    run_split_lab_v4_mimic(args.model, "dynamic_int8", "./output", 128, 1280)
