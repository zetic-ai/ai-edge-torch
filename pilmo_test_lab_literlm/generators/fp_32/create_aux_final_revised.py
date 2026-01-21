import os
import torch
import torch.nn as nn
import ai_edge_torch
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from common_config import Gemma3Config
from export_utils import export_and_compile


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


def main():
    tflite_path = os.path.join(Gemma3Config.OUTPUT_BIN_DIR, "gemma3_1b_aux.tflite")
    conv = ai_edge_torch._convert.converter.Converter()
    add_aux_signatures(
        conv,
        Gemma3CacheUpdate().eval(),
        Gemma3MaskGenerator(mode="decode").eval(),
        Gemma3MaskGenerator(mode="prefill").eval(),
        Gemma3RoPEOfficial().eval(),
    )

    export_and_compile(
        conv, tflite_path, aot_signatures=["decode_mask", "prefill_mask_128"]
    )


if __name__ == "__main__":
    main()
