import math
import os

import ai_edge_litert.aot.core.components as aot_components
import ai_edge_litert.aot.vendors.qualcomm.qualcomm_backend as qnn_backend
import numpy as np
import torch
import torch.nn as nn
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import apply_plugin
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder


# ==============================================================================
# MONKEYPATCH: FORCE QUALCOMM WEIGHT SHARING
# ==============================================================================
@qnn_backend._call_component.register(apply_plugin.ApplyPlugin)
def patched_apply_plugin(
    component: aot_components.ApplyPluginT,
    backend: qnn_backend.QualcommBackend,
    input_model,
    output_model,
):
    import os
    import pathlib

    from ai_edge_litert.aot.core import common

    plugin_path = common.get_resource(
        pathlib.Path("vendors/qualcomm/compiler/libLiteRtCompilerPlugin_Qualcomm.so")
    )
    lib_dir = os.path.dirname(plugin_path)
    try:
        import ai_edge_litert_sdk_qualcomm

        sdk_libs_path = str(ai_edge_litert_sdk_qualcomm.path_to_sdk_libs())
    except ImportError:
        sdk_libs_path = None

    extra_kwargs = {
        "libs": lib_dir,
        "sdk_libs_path": sdk_libs_path,
        "qualcomm_enable_weight_sharing": "true",  # THIS IS THE KEY
    }
    print(f"DEBUG: Calling ApplyPlugin with {extra_kwargs}")
    return component(
        input_model,
        output_model,
        backend.soc_manufacturer,
        backend.soc_model,
        **extra_kwargs,
    )


# ==============================================================================
# MODEL CONFIG
# ==============================================================================
NUM_LAYERS = 26
KV_CACHE_LEN = 1280
PREFILL_T = 128
HEAD_DIM = 256
EMBED_DIM = 1152
VOCAB_SIZE = 262144


class Gemma3Main(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer_blocks = model.transformer_blocks
        self.final_norm = model.final_norm
        self.lm_head = model.lm_head

    def forward(
        self,
        embeddings,
        mask_global,
        mask_local,
        pos_emb_cos,
        pos_emb_sin,
        pos_emb_local_cos,
        pos_emb_local_sin,
        **kwargs,
    ):
        h = embeddings
        output_data = {}
        for i in range(NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c = pos_emb_local_cos if is_local else pos_emb_cos
            l_rope_s = pos_emb_local_sin if is_local else pos_emb_sin
            l_mask = mask_local if is_local else mask_global

            x = self.rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)
            B, T, _ = qkv.shape
            ng, nh, dim = (
                block.atten_func.config.num_query_groups,
                block.atten_func.config.num_heads,
                block.atten_func.config.head_dim,
            )
            group_size = nh // ng
            qkv = qkv.view(B, T, ng, (group_size + 2), dim)
            q, k, v = qkv.split([group_size, 1, 1], dim=-2)
            q, k, v = (
                q.reshape(B, T, nh, dim),
                k.reshape(B, T, ng, dim),
                v.reshape(B, T, ng, dim),
            )

            q = self.rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = self.rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            q, k = (
                self.apply_rope(q, l_rope_c, l_rope_s),
                self.apply_rope(k, l_rope_c, l_rope_s),
            )

            k_new, v_new = k.transpose(1, 2), v.transpose(1, 2).transpose(2, 3)
            output_data[f"kv_slice_k_{i}"], output_data[f"kv_slice_v_{i}"] = (
                k_new,
                v_new,
            )
            k_cache, v_cache = kwargs[f"kv_cache_k_{i}"], kwargs[f"kv_cache_v_{i}"]
            k_full, v_full = (
                torch.cat([k_cache, k_new], dim=2),
                torch.cat([v_cache, v_new], dim=3),
            )

            q_a = q.permute(0, 2, 1, 3)
            if nh > ng:
                k_full, v_full = (
                    k_full.repeat_interleave(nh // ng, dim=1),
                    v_full.repeat_interleave(nh // ng, dim=1),
                )

            scores = (q_a @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
            if block.atten_func.config.logit_softcap:
                sc = block.atten_func.config.logit_softcap
                scores = torch.tanh(scores / sc) * sc
            scores = scores + l_mask
            probs = torch.softmax(scores.float(), dim=-1).type_as(q)
            attn_out = (
                (probs @ v_full.transpose(-2, -1)).transpose(1, 2).reshape(B, T, -1)
            )

            y = block.atten_func.output_projection(attn_out)
            h = h + self.rms_norm(y, block.post_atten_norm.weight)
            h = h + block.ff(h)

        # Logits only for Decode (T=1)
        if embeddings.shape[1] == 1:
            h = self.rms_norm(h, self.final_norm.weight)
            logits = self.lm_head(h)
            output_data["logits"] = logits

        return output_data

    def rms_norm(self, x, weight, eps=1e-6, zero_centered=True):
        if weight is None:
            return x
        norm_x = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm_x + eps)
        if zero_centered:
            return (x_normed * (1.0 + weight)).type_as(x)
        return (x_normed * weight).type_as(x)

    def apply_rope(self, x, cos, sin):
        d = x.shape[-1]
        x_left = x[..., : d // 2]
        x_right = x[..., d // 2 :]
        x_rotated = torch.cat([-x_right, x_left], dim=-1)
        return (x * cos) + (x_rotated * sin)


def generate_samples(num_samples=5, T=1, to_numpy=True):
    samples = []
    mask_len = KV_CACHE_LEN + T
    with torch.no_grad():
        for _ in range(num_samples):
            sample = {
                "embeddings": torch.randn((1, T, EMBED_DIM)),
                "mask_global": torch.zeros((1, 1, T, mask_len)),
                "mask_local": torch.zeros((1, 1, T, mask_len)),
                "pos_emb_cos": torch.randn((1, T, 1, HEAD_DIM)),
                "pos_emb_sin": torch.randn((1, T, 1, HEAD_DIM)),
                "pos_emb_local_cos": torch.randn((1, T, 1, HEAD_DIM)),
                "pos_emb_local_sin": torch.randn((1, T, 1, HEAD_DIM)),
            }
            for i in range(NUM_LAYERS):
                sample[f"kv_cache_k_{i}"] = (
                    torch.randn((1, 1, KV_CACHE_LEN, HEAD_DIM)) * 0.1
                )
                sample[f"kv_cache_v_{i}"] = (
                    torch.randn((1, 1, HEAD_DIM, KV_CACHE_LEN)) * 0.1
                )
            if to_numpy:
                sample = {
                    k: v.numpy() if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()
                }
            samples.append(sample)
    return samples


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)

    # --- [1/2] MAIN MODEL ---
    print("\n--- [1/2] MAIN MODEL (W4A16, Unified Signatures, WeightSharing) ---")
    tflite_path = os.path.join(output_dir, "optimized_main_v3_fp32.tflite")
    w4a16_path = os.path.join(output_dir, "optimized_main_v3_w4a16.tflite")
    aot_path = os.path.join(output_dir, "optimized_main_v3_w4a16_aot.tflite")

    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    main_mod = Gemma3Main(decoder.Decoder(config)).eval()

    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "decode", main_mod, sample_kwargs=generate_samples(1, T=1, to_numpy=False)[0]
    )
    conv.add_signature(
        "prefill_128",
        main_mod,
        sample_kwargs=generate_samples(1, T=PREFILL_T, to_numpy=False)[0],
    )
    edge_model = conv.convert()
    edge_model.export(tflite_path)

    qt = quantizer.Quantizer(float_model=tflite_path)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=op_config,
    )

    calib_data = {
        "decode": generate_samples(num_samples=2, T=1, to_numpy=True),
        "prefill_128": generate_samples(num_samples=2, T=PREFILL_T, to_numpy=True),
    }
    res = qt.calibrate(calib_data)
    quant_result = qt.quantize(res)
    quant_result.export_model(w4a16_path, overwrite=True)

    print("AOT Compiling Main (Monkeypatched Weight Sharing)...")
    litert_model = litert_types.Model.create_from_bytes(quant_result.quantized_model)
    target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
    aot_config = [litert_types.CompilationConfig(target=target)]
    aot_res = aot_lib.aot_compile(litert_model, config=aot_config)
    if aot_res.models_with_backend:
        with open(aot_path, "wb") as f:
            f.write(aot_res.models_with_backend[0][1].model_bytes)
        print(f"✅ Main AOT model saved at {aot_path}")

    # --- [2/2] EMBEDDER ---
    print("\n--- [2/2] EMBEDDER (W4 Optimized) ---")
    embed_fp32 = os.path.join(output_dir, "optimized_embedder_v3_fp32.tflite")
    embed_w4 = os.path.join(output_dir, "optimized_embedder_v3_w4.tflite")

    class EmbedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

        def forward(self, input):
            return self.emb(input)

    embed_mod = EmbedModel().eval()
    conv_e = ai_edge_torch._convert.converter.Converter()
    conv_e.add_signature(
        "decode_embedder",
        embed_mod,
        sample_kwargs={"input": torch.zeros((1, 1), dtype=torch.int32)},
    )
    conv_e.add_signature(
        "prefill_embedder_128",
        embed_mod,
        sample_kwargs={"input": torch.zeros((1, 128), dtype=torch.int32)},
    )
    edge_embed = conv_e.convert()
    edge_embed.export(embed_fp32)

    qt_e = quantizer.Quantizer(float_model=embed_fp32)
    # W4 Weight-Only using ComputePrecision.INTEGER on EMBEDDING_LOOKUP
    op_config_e = qtyping.OpQuantizationConfig(
        activation_tensor_config=None,
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    qt_e.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP,
        op_config=op_config_e,
    )

    calib_e = {
        "decode_embedder": [{"input": np.zeros((1, 1), dtype=np.int32)}],
        "prefill_embedder_128": [{"input": np.zeros((1, 128), dtype=np.int32)}],
    }
    res_e = qt_e.calibrate(calib_e)
    quant_e = qt_e.quantize(res_e)
    quant_e.export_model(embed_w4, overwrite=True)
    print(f"✅ Embedder W4 saved at {embed_w4}")

    # Packaging
    from ai_edge_litert.internal import litertlm_builder

    final_path = os.path.join(output_dir, "gemma3_1b_w4a16_v3_final.litertlm")
    tokenizer_path = os.path.join(output_dir, "tokenizer.model")
    metadata_path = os.path.join(output_dir, "llm_metadata.pb")
    aux_model = os.path.join(output_dir, "auxiliary.tflite")

    builder = litertlm_builder.LitertLmFileBuilder()
    builder.add_system_metadata(
        litertlm_builder.Metadata(
            key="Authors", value="Zetic.ai", dtype=litertlm_builder.DType.STRING
        )
    )
    builder.add_llm_metadata(metadata_path)
    builder.add_sentencepiece_tokenizer(tokenizer_path)
    builder.add_tflite_model(aot_path, litertlm_builder.TfLiteModelType.PREFILL_DECODE)
    builder.add_tflite_model(embed_w4, litertlm_builder.TfLiteModelType.EMBEDDER)
    builder.add_tflite_model(aux_model, litertlm_builder.TfLiteModelType.AUX)
    with open(final_path, "wb") as f:
        builder.build(f)
    print(f"\n🎉 FINAL PACKAGE: {final_path}")
    print(f"  Main AOT: {os.path.getsize(aot_path) / 1e6:.2f} MB")
    print(f"  Embedder W4: {os.path.getsize(embed_w4) / 1e6:.2f} MB")
    print(f"  Total Package: {os.path.getsize(final_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
