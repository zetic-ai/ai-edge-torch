import os
import torch
import torch.nn as nn
import ai_edge_torch
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

# Shared Constants
NUM_LAYERS = 26
HEAD_DIM = 256
KV_MAX_LEN = 1280
SLIDING_WINDOW = 512


# 1. Gemma3CacheUpdate (Copied EXACTLY from create_cache_update_aot.py)
class Gemma3CacheUpdate(nn.Module):
    def __init__(self, layer_indices=range(26), kv_cache_max_len=1280):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.kv_cache_max_len = kv_cache_max_len

    def forward(self, input_pos: torch.Tensor, **kwargs):
        """
        Gemma3 Cache Update: Exact Full Graph Reproduction.
        - Covers all layers from 0 to 25.
        - input_pos is expected as a tensor of shape [1] or [T].
        - Replicates Cast -> Concat -> Cast pattern for DynamicUpdateSlice indices.
        """
        results = {}

        # 1. Cast input_pos to float32 (Using first element for slice start)
        pos_f = input_pos[0:1].float()  # [1]
        z_f = torch.zeros([1], dtype=torch.float32)

        # 2. Replicate the Concatenation to [4] float32
        # K coordinates: [batch=0, head=0, seq=pos, head_dim=0]
        k_idx_f = torch.cat([z_f, z_f, pos_f, z_f], dim=0)
        k_idx_i = k_idx_f.int()  # Cast to int32[4]

        # V coordinates: [batch=0, head=0, head_dim=0, seq=pos] (Transposed V)
        v_idx_f = torch.cat([z_f, z_f, z_f, pos_f], dim=0)
        v_idx_i = v_idx_f.int()  # Cast to int32[4]

        # Provide the coordinate list to custom DUS op
        k_list = [k_idx_i[0], k_idx_i[1], k_idx_i[2], k_idx_i[3]]
        v_list = [v_idx_i[0], v_idx_i[1], v_idx_i[2], v_idx_i[3]]

        for i in self.layer_indices:
            # Update K: [1, 1, 1280, 256]
            k_cache_key = f"kv_cache_k_{i}"
            k_slice_key = f"kv_slice_k_{i}"
            if k_cache_key in kwargs and k_slice_key in kwargs:
                results[k_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[k_cache_key], kwargs[k_slice_key], k_list
                )

            # Update V: [1, 1, 256, 1280] (Transposed)
            v_cache_key = f"kv_cache_v_{i}"
            v_slice_key = f"kv_slice_v_{i}"
            if v_cache_key in kwargs and v_slice_key in kwargs:
                results[v_cache_key] = dus_utils.dynamic_update_slice(
                    kwargs[v_cache_key], kwargs[v_slice_key], v_list
                )

        return results


# 2. Gemma3MaskGenerator (Copied EXACTLY from create_decode_mask_aot_final.py)
class Gemma3MaskGenerator(nn.Module):
    def __init__(self, mode="decode"):
        super().__init__()
        self.mode = mode
        self.mask_val = -1e4

    def forward(self, time_step: torch.Tensor, input_tokens: torch.Tensor):
        # time_step: Scalar [ ]
        # input_tokens: [1, T] - Used for graph connectivity and seq_len extraction
        gate = (input_tokens.float().mean() > -1e9).float()
        seq_len = input_tokens.shape[1]

        # 1. Coordinate Generation
        if self.mode == "decode":
            curr_kv_len = 1281
            eff_q_pos = time_step.view(1)  # [1]
        else:
            curr_kv_len = 1408
            # For prefill, eff_q_pos is [seq_len]
            eff_q_pos = time_step + torch.arange(seq_len, device=input_tokens.device)

        kv_idx = torch.arange(curr_kv_len, device=input_tokens.device).view(1, 1, 1, -1)

        # 2. Mask Logic (Broadcast to [1, 1, T, KV_LEN])
        is_causal = kv_idx <= eff_q_pos.view(1, 1, -1, 1)
        is_window = kv_idx > (eff_q_pos.view(1, 1, -1, 1) - SLIDING_WINDOW)

        mg = torch.where(is_causal, 0.0, self.mask_val) * gate
        ml = torch.where(is_causal & is_window, 0.0, self.mask_val) * gate

        return {"mask_global": mg, "mask_local": ml}


# 3. Gemma3RoPEOfficial (Copied EXACTLY from create_rope_aot.py)
class Gemma3RoPEOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-compute inverse frequencies for Global (10k) and Local (1M)
        inv_freq_g = 1.0 / (10000 ** (torch.arange(0, 256, 2).float() / 256))
        inv_freq_l = 1.0 / (1000000 ** (torch.arange(0, 256, 2).float() / 256))
        self.register_buffer("inv_freq_g", inv_freq_g)
        self.register_buffer("inv_freq_l", inv_freq_l)

    def compute_branch(self, pos_reshaped, inv_freq):
        # 1. Position-frequency calculation: [1, T, 1] * [1, 1, 128] -> [1, T, 128]
        freqs = pos_reshaped * inv_freq.view(1, 1, -1)

        # 2. Sinusoidal functions
        c = torch.cos(freqs)
        s = torch.sin(freqs)

        # 3. Reshape to official 4D format [1, 1, T, 128]
        t = pos_reshaped.shape[1]
        c_4d = c.view(1, 1, t, 128)
        s_4d = s.view(1, 1, t, 128)

        # 4. Negate Sin branch for rotation parity: x*cos - y*sin
        s_v_neg = s_4d * -1.0

        # 5. Concatenate to full head dimension [1, 1, T, 256]
        return torch.cat([c_4d, c_4d], dim=-1), torch.cat([s_v_neg, s_v_neg], dim=-1)

    def forward(self, input_pos: torch.Tensor):
        # input_pos: int32[T] (T=128 for prefill, T=1 for decode)
        t = input_pos.shape[0]
        # Entry Reshape matching official spec
        pos_reshaped = input_pos.view(1, t, 1).float()

        cos_g, sin_g = self.compute_branch(pos_reshaped, self.inv_freq_g)
        cos_l, sin_l = self.compute_branch(pos_reshaped, self.inv_freq_l)

        return {
            "pos_emb_cos": cos_g,
            "pos_emb_sin": sin_g,
            "pos_emb_local_cos": cos_l,
            "pos_emb_local_sin": sin_l,
        }


def main():
    output_dir = (
        "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/bin/main_int8"
    )
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_aux.tflite")
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")

    # Instances
    mask_dec = Gemma3MaskGenerator(mode="decode").eval()
    mask_pre = Gemma3MaskGenerator(mode="prefill").eval()
    rope_mod = Gemma3RoPEOfficial().eval()
    update_mod = Gemma3CacheUpdate().eval()

    conv = ai_edge_torch._convert.converter.Converter()

    # 0. decode_cache_update
    kwargs_dec_upd = {"input_pos": torch.zeros(1, dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        kwargs_dec_upd[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, 1280, 256), dtype=torch.float32
        )
        kwargs_dec_upd[f"kv_slice_k_{i}"] = torch.zeros(
            (1, 1, 1, 256), dtype=torch.float32
        )
        kwargs_dec_upd[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, 256, 1280), dtype=torch.float32
        )
        kwargs_dec_upd[f"kv_slice_v_{i}"] = torch.zeros(
            (1, 1, 256, 1), dtype=torch.float32
        )
    conv.add_signature("decode_cache_update", update_mod, sample_kwargs=kwargs_dec_upd)

    # 1. decode_mask
    conv.add_signature(
        "decode_mask",
        mask_dec,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 1), dtype=torch.int32),
        },
    )

    # 2. decode_rope
    conv.add_signature(
        "decode_rope",
        rope_mod,
        sample_kwargs={"input_pos": torch.zeros(1, dtype=torch.int32)},
    )

    # 3. prefill_cache_update_128
    kwargs_pre_upd = {"input_pos": torch.zeros(128, dtype=torch.int32)}
    for i in range(NUM_LAYERS):
        kwargs_pre_upd[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, 1280, 256), dtype=torch.float32
        )
        kwargs_pre_upd[f"kv_slice_k_{i}"] = torch.zeros(
            (1, 1, 128, 256), dtype=torch.float32
        )
        kwargs_pre_upd[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, 256, 1280), dtype=torch.float32
        )
        kwargs_pre_upd[f"kv_slice_v_{i}"] = torch.zeros(
            (1, 1, 256, 128), dtype=torch.float32
        )
    conv.add_signature(
        "prefill_cache_update_128", update_mod, sample_kwargs=kwargs_pre_upd
    )

    # 4. prefill_mask_128
    conv.add_signature(
        "prefill_mask_128",
        mask_pre,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 128), dtype=torch.int32),
        },
    )

    # 5. prefill_rope_128
    conv.add_signature(
        "prefill_rope_128",
        rope_mod,
        sample_kwargs={"input_pos": torch.zeros(128, dtype=torch.int32)},
    )

    print("Converting Integrated Gemma3 Aux Model (Strict Parity Mode)...")
    edge_model = conv.convert()
    edge_model.export(tflite_path)

    # Selective AOT: Only Mask Signatures
    print("Starting AOT compilation for MASK signatures only...")
    try:
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)

        aot_sigs = ["decode_mask", "prefill_mask_128"]
        config = [
            litert_types.CompilationConfig(target=target, signature_names=aot_sigs)
        ]

        result = aot_lib.aot_compile(litert_model, config=config)
        if result.models_with_backend:
            with open(aot_tflite_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(f"SUCCESS: Integrated AOT Model saved at {aot_tflite_path}")
            print(f"AOT targets (NPU): {aot_sigs}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
