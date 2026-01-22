import torch
import torch.nn as nn
import math
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from common_config import Gemma3Config


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
        self.config = model.config
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
        head_dim = Gemma3Config.HEAD_DIM

        for i in range(Gemma3Config.NUM_LAYERS):
            block = self.transformer_blocks[i]
            is_local = (i + 1) % 6 == 0
            l_rope_c, l_rope_s = (
                (pos_emb_local_cos, pos_emb_local_sin)
                if is_local
                else (pos_emb_cos, pos_emb_sin)
            )
            l_mask = mask_local if is_local else mask_global

            # --- 1. Attention Branch ---
            # Pre-attention norm and QKV projection
            x = rms_norm(h, block.pre_atten_norm.weight)
            qkv = block.atten_func.qkv_projection(x)

            B, T, _ = qkv.shape
            ng = block.atten_func.config.num_query_groups
            nh = block.atten_func.config.num_heads
            q_per_kv = nh // ng

            # Reshape/Split (NPU-friendly view pattern)
            qkv = qkv.view(B, T, ng, q_per_kv + 2, head_dim)
            q, k, v = qkv.split([q_per_kv, 1, 1], dim=-2)

            q = q.reshape(B, T, nh, head_dim)
            k = k.reshape(B, T, ng, head_dim)
            v = v.reshape(B, T, ng, head_dim)

            # Optional Q/K normalization
            q = rms_norm(q, getattr(block.atten_func.query_norm, "weight", None))
            k = rms_norm(k, getattr(block.atten_func.key_norm, "weight", None))

            # Apply RoPE
            q = apply_rope(q, l_rope_c, l_rope_s)
            k = apply_rope(k, l_rope_c, l_rope_s)

            # --- Pure NPU KV Cache Management ---
            # Official model returns the NEW slice, does NOT update the full buffer in-place.
            # k_new shape: [B, ng, 1, head_dim] -> [1, 1, 1, 256]
            # v_new shape: [B, ng, head_dim, 1] -> [1, 1, 256, 1] (Transposed)
            k_new = k.transpose(1, 2)
            v_new = v.transpose(1, 2).transpose(2, 3)

            output_data[f"kv_slice_k_{i}"] = k_new
            output_data[f"kv_slice_v_{i}"] = v_new

            # Load history buffers from inputs (provided by runtime/aux model)
            # kv_cache_k_{i} shape: [1, 1, 1280, 256]
            # kv_cache_v_{i} shape: [1, 1, 256, 1280]
            k_history = kwargs[f"kv_cache_k_{i}"]
            v_history = kwargs[f"kv_cache_v_{i}"]

            # Concatenate history with new token for current attention calculation
            # k_full: [1, 1, 1281, 256]
            # v_full: [1, 1, 256, 1281]
            k_full = torch.cat([k_history, k_new], dim=2)
            v_full = torch.cat([v_history, v_new], dim=3)

            # --- SDPA with Implicit GQA Broadcasting ---
            q_a = q.permute(0, 2, 1, 3)  # [B, nh, 1, head_dim]

            # Score: [B, nh, 1, 1281]
            scores = (q_a @ k_full.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

            if block.atten_func.config.logit_softcap:
                sc = block.atten_func.config.logit_softcap
                scores = torch.tanh(scores / sc) * sc

            scores = scores + l_mask

            # Attn: [B, nh, 1, 1281]
            attn = torch.softmax(scores.float(), dim=-1).type_as(q)

            # Output: [B, nh, 1, head_dim]
            # (B, nh, 1, 1281) @ (B, 1, 1281, head_dim) -> (B, nh, 1, head_dim)
            attn_out = attn @ v_full.transpose(-2, -1)

            # Output Projection and Residual connection
            attn_out = block.atten_func.output_projection(
                attn_out.transpose(1, 2).reshape(B, T, -1)
            )
            h = h + rms_norm(attn_out, block.post_atten_norm.weight)

            # --- 2. Feed-Forward Branch (MLP Flattening) ---
            # Manually implement MLP to ensure all layers are visible to the compiler
            x_ff = rms_norm(h, block.ff.pre_ff_norm.weight)

            # Gated MLP pattern: (act(w1(x)) * w3(x)) -> w2
            w1_out = block.ff.w1(x_ff)
            w3_out = block.ff.w3(x_ff)
            ff_h = torch.nn.functional.gelu(w1_out, approximate="tanh") * w3_out
            ff_out = block.ff.w2(ff_h)

            # Final FF residual connection with post-norm
            h = h + rms_norm(ff_out, block.ff.post_ff_norm.weight)

        # Output Head
        h = rms_norm(h, self.final_norm.weight)
        output_data["logits"] = self.lm_head(h)
        return output_data


class Gemma3Embedder(nn.Module):
    def __init__(
        self, vocab_size=Gemma3Config.VOCAB_SIZE, embedding_dim=Gemma3Config.EMBED_DIM
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids: torch.Tensor):
        return self.embedding(token_ids)


class Gemma3EmbedderSignatureWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, token_ids: torch.Tensor):
        return {"embeddings": self.model(token_ids)}


class Gemma3CacheUpdate(nn.Module):
    def __init__(
        self,
        layer_indices=range(Gemma3Config.NUM_LAYERS),
        kv_cache_max_len=Gemma3Config.KV_CACHE_LEN,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.kv_cache_max_len = kv_cache_max_len

    def forward(self, input_pos: torch.Tensor, **kwargs):
        results = {}
        pos_f = input_pos[0:1].float()
        z_f = torch.zeros([1], dtype=torch.float32)

        k_idx_f = torch.cat([z_f, z_f, pos_f, z_f], dim=0)
        k_idx_i = k_idx_f.int()
        v_idx_f = torch.cat([z_f, z_f, z_f, pos_f], dim=0)
        v_idx_i = v_idx_f.int()

        k_list = [k_idx_i[0], k_idx_i[1], k_idx_i[2], k_idx_i[3]]
        v_list = [v_idx_i[0], v_idx_i[1], v_idx_i[2], v_idx_i[3]]

        for i in self.layer_indices:
            k_cache_key, k_slice_key = f"kv_cache_k_{i}", f"kv_slice_k_{i}"
            if k_cache_key in kwargs and k_slice_key in kwargs:
                results[k_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[k_cache_key], kwargs[k_slice_key], k_list
                )

            v_cache_key, v_slice_key = f"kv_cache_v_{i}", f"kv_slice_v_{i}"
            if v_cache_key in kwargs and v_slice_key in kwargs:
                results[v_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[v_cache_key], kwargs[v_slice_key], v_list
                )
        return results


class Gemma3MaskGenerator(nn.Module):
    def __init__(self, mode="decode"):
        super().__init__()
        self.mode = mode
        self.mask_val = -1e4

    def forward(self, time_step: torch.Tensor, input_tokens: torch.Tensor):
        gate = (input_tokens.float().mean() > -1e9).float()
        seq_len = input_tokens.shape[1]

        if self.mode == "decode":
            curr_kv_len = Gemma3Config.DECODE_MASK_LEN
            eff_q_pos = time_step.view(1)
        else:
            curr_kv_len = Gemma3Config.PREFILL_MASK_LEN
            eff_q_pos = time_step + torch.arange(seq_len, device=input_tokens.device)

        kv_idx = torch.arange(curr_kv_len, device=input_tokens.device).view(1, 1, 1, -1)
        is_causal = kv_idx <= eff_q_pos.view(1, 1, -1, 1)
        is_window = kv_idx > (eff_q_pos.view(1, 1, -1, 1) - Gemma3Config.SLIDING_WINDOW)

        mg = torch.where(is_causal, 0.0, self.mask_val) * gate
        ml = torch.where(is_causal & is_window, 0.0, self.mask_val) * gate
        return {"mask_global": mg, "mask_local": ml}


class Gemma3RoPEOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq_g = 1.0 / (
            10000
            ** (
                torch.arange(0, Gemma3Config.HEAD_DIM, 2).float()
                / Gemma3Config.HEAD_DIM
            )
        )
        inv_freq_l = 1.0 / (
            1000000
            ** (
                torch.arange(0, Gemma3Config.HEAD_DIM, 2).float()
                / Gemma3Config.HEAD_DIM
            )
        )
        self.register_buffer("inv_freq_g", inv_freq_g)
        self.register_buffer("inv_freq_l", inv_freq_l)

    def compute_branch(self, pos_reshaped, inv_freq):
        freqs = pos_reshaped * inv_freq.view(1, 1, -1)
        c, s = torch.cos(freqs), torch.sin(freqs)
        t = pos_reshaped.shape[1]
        c_4d, s_4d = c.view(1, 1, t, 128), s.view(1, 1, t, 128)
        return torch.cat([c_4d, c_4d], dim=-1), torch.cat(
            [s_4d * -1.0, s_4d * -1.0], dim=-1
        )

    def forward(self, input_pos: torch.Tensor):
        t = input_pos.shape[0]
        pos_reshaped = input_pos.view(1, t, 1).float()
        cos_g, sin_g = self.compute_branch(pos_reshaped, self.inv_freq_g)
        cos_l, sin_l = self.compute_branch(pos_reshaped, self.inv_freq_l)
        return {
            "pos_emb_cos": cos_g,
            "pos_emb_sin": sin_g,
            "pos_emb_local_cos": cos_l,
            "pos_emb_local_sin": sin_l,
        }


def add_aux_signatures(conv, update_mod, mask_dec, mask_pre, rope_mod):
    # Decode Signatures
    kwargs_dec_upd = {"input_pos": torch.zeros(1, dtype=torch.int32)}
    for i in range(Gemma3Config.NUM_LAYERS):
        kwargs_dec_upd[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.KV_CACHE_LEN, Gemma3Config.HEAD_DIM),
            dtype=torch.float32,
        )
        kwargs_dec_upd[f"kv_slice_k_{i}"] = torch.zeros(
            (1, 1, 1, Gemma3Config.HEAD_DIM), dtype=torch.float32
        )
        kwargs_dec_upd[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.KV_CACHE_LEN),
            dtype=torch.float32,
        )
        kwargs_dec_upd[f"kv_slice_v_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.HEAD_DIM, 1), dtype=torch.float32
        )

    conv.add_signature("decode_cache_update", update_mod, sample_kwargs=kwargs_dec_upd)
    conv.add_signature(
        "decode_mask",
        mask_dec,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 1), dtype=torch.int32),
        },
    )
    conv.add_signature(
        "decode_rope",
        rope_mod,
        sample_kwargs={"input_pos": torch.zeros(1, dtype=torch.int32)},
    )

    # Prefill Signatures
    kwargs_pre_upd = {
        "input_pos": torch.zeros(Gemma3Config.PREFILL_T, dtype=torch.int32)
    }
    for i in range(Gemma3Config.NUM_LAYERS):
        kwargs_pre_upd[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.KV_CACHE_LEN, Gemma3Config.HEAD_DIM),
            dtype=torch.float32,
        )
        kwargs_pre_upd[f"kv_slice_k_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.PREFILL_T, Gemma3Config.HEAD_DIM), dtype=torch.float32
        )
        kwargs_pre_upd[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.KV_CACHE_LEN),
            dtype=torch.float32,
        )
        kwargs_pre_upd[f"kv_slice_v_{i}"] = torch.zeros(
            (1, 1, Gemma3Config.HEAD_DIM, Gemma3Config.PREFILL_T), dtype=torch.float32
        )

    conv.add_signature(
        "prefill_cache_update_128", update_mod, sample_kwargs=kwargs_pre_upd
    )
    conv.add_signature(
        "prefill_mask_128",
        mask_pre,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, Gemma3Config.PREFILL_T), dtype=torch.int32),
        },
    )
    conv.add_signature(
        "prefill_rope_128",
        rope_mod,
        sample_kwargs={
            "input_pos": torch.zeros(Gemma3Config.PREFILL_T, dtype=torch.int32)
        },
    )
