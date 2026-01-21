import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

# Official Configuration for Gemma3-1B
HEAD_DIM = 256
HALF_DIM = 128


class Gemma3RoPEOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-compute inverse frequencies for Global (10k) and Local (1M)
        inv_freq_g = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        inv_freq_l = 1.0 / (
            1000000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        )
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
        c_4d = c.view(1, 1, t, HALF_DIM)
        s_4d = s.view(1, 1, t, HALF_DIM)

        # 4. Negate Sin branch for rotation parity: x*cos - y*sin
        # This matches the "Mul -1" nodes found in official TFLite models
        s_v_neg = s_4d * -1.0

        # 5. Concatenate to full head dimension [1, 1, T, 256]
        # Replicating the [cos, cos] and [-sin, -sin] structure
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
    output_dir = "./pilmo_test_lab_literlm/bin"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_rope_aot.tflite")
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")

    rope_mod = Gemma3RoPEOfficial().eval()

    print("Exporting Official Gemma3 RoPE (Clean Implementation)...")
    conv = ai_edge_torch._convert.converter.Converter()

    # Multi-Signature Support
    conv.add_signature(
        "prefill_128",
        rope_mod,
        sample_kwargs={"input_pos": torch.zeros(128, dtype=torch.int32)},
    )
    conv.add_signature(
        "decode",
        rope_mod,
        sample_kwargs={"input_pos": torch.zeros(1, dtype=torch.int32)},
    )

    edge_model = conv.convert()
    edge_model.export(tflite_path)

    print("AOT Compiling (SM8750)...")
    try:
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()
        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)
        config = [litert_types.CompilationConfig(target=target)]
        result = aot_lib.aot_compile(litert_model, config=config)
        if result.models_with_backend:
            with open(aot_tflite_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(f"SUCCESS: Clean Official RoPE AOT Model saved at {aot_tflite_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
