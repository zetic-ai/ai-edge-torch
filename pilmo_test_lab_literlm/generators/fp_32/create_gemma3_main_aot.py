import os
import torch
import torch.nn as nn
import math
import ai_edge_torch
from ai_edge_torch.generative.examples.gemma3 import decoder
from common_config import Gemma3Config
from export_utils import export_and_compile


def rms_norm(x, weight, eps=1e-6, zero_centered=True):
    if weight is None:
        return x
    norm_x = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x.float() * torch.rsqrt(norm_x + eps)
    if zero_centered:
        return (x_normed * (1.0 + weight)).type_as(x)
    return (x_normed * weight).type_as(x)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x_rotated = torch.cat([-x[..., d // 2 :], x[..., : d // 2]], dim=-1)
    return (x * cos) + (x_rotated * sin)


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

        for i in range(Gemma3Config.NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c, l_rope_s = (
                (pos_emb_local_cos, pos_emb_local_sin)
                if is_local
                else (pos_emb_cos, pos_emb_sin)
            )
            l_mask = mask_local if is_local else mask_global

            x = rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)

            B, T, ng, nh, dim = (
                qkv.shape[0],
                qkv.shape[1],
                block.atten_func.config.num_query_groups,
                block.atten_func.config.num_heads,
                block.atten_func.config.head_dim,
            )
            qkv = qkv.view(B, T, ng, (nh // ng + 2), dim)
            q, k, v = qkv.split([nh // ng, 1, 1], dim=-2)
            q, k, v = (
                q.reshape(B, T, nh, dim),
                k.reshape(B, T, ng, dim),
                v.reshape(B, T, ng, dim),
            )

            q = rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))
            q, k = apply_rope(q, l_rope_c, l_rope_s), apply_rope(k, l_rope_c, l_rope_s)

            k_new, v_new = k.transpose(1, 2), v.transpose(1, 2).transpose(2, 3)
            output_data[f"kv_slice_k_{i}"], output_data[f"kv_slice_v_{i}"] = (
                k_new,
                v_new,
            )

            k_full, v_full = (
                torch.cat([kwargs[f"kv_cache_k_{i}"], k_new], dim=2),
                torch.cat([kwargs[f"kv_cache_v_{i}"], v_new], dim=3),
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
            attn_out = torch.softmax(scores.float(), dim=-1).type_as(
                q
            ) @ v_full.transpose(-2, -1)
            h = h + rms_norm(
                block.atten_func.output_projection(
                    attn_out.transpose(1, 2).reshape(B, T, -1)
                ),
                block.post_atten_norm.weight,
            )
            h = h + block.ff(h)

        h = rms_norm(h, self.final_norm.weight)
        output_data["logits"] = self.lm_head(h)
        return output_data


def get_sample_kwargs(t_len, mask_len):
    kwargs = {
        "embeddings": torch.zeros(
            (1, t_len, Gemma3Config.EMBED_DIM), dtype=torch.float32
        ),
        "mask_global": torch.zeros((1, 1, t_len, mask_len), dtype=torch.float32),
        "mask_local": torch.zeros((1, 1, t_len, mask_len), dtype=torch.float32),
        "pos_emb_cos": torch.zeros(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_sin": torch.zeros(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_local_cos": torch.zeros(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
        "pos_emb_local_sin": torch.zeros(
            (1, t_len, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        ),
    }
    for i in range(Gemma3Config.NUM_LAYERS):
        kwargs[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.KV_CACHE_LEN, Gemma3Config.HEAD_DIM),
            dtype=torch.float32,
        )
        kwargs[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.KV_CACHE_LEN),
            dtype=torch.float32,
        )
    return kwargs


def main():
    tflite_path = os.path.join(Gemma3Config.MAIN_BIN_DIR, "gemma3_1b_main.tflite")
    config = decoder.get_decoder_config_1b()
    config.mask_cache_size = 0
    main_mod = Gemma3Main(decoder.Decoder(config)).eval()

    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "decode",
        main_mod,
        sample_kwargs=get_sample_kwargs(1, Gemma3Config.DECODE_MASK_LEN),
    )
    conv.add_signature(
        "prefill_128",
        main_mod,
        sample_kwargs=get_sample_kwargs(
            Gemma3Config.PREFILL_T, Gemma3Config.PREFILL_MASK_LEN
        ),
    )

    export_and_compile(
        conv, tflite_path, aot_signatures=None
    )  # Compile all signatures for main


if __name__ == "__main__":
    main()
