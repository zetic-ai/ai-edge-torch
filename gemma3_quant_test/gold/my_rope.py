import os

import numpy as np
import torch
import torch.nn as nn
from ai_edge_quantizer import qtyping, quantizer

import ai_edge_torch

# Official Configuration for Gemma3-1B
HEAD_DIM = 256
HALF_DIM = 128


class Gemma3RoPEOfficial(nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq_g = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
        inv_freq_l = 1.0 / (
            1000000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
        )
        self.register_buffer("inv_freq_g", inv_freq_g)
        self.register_buffer("inv_freq_l", inv_freq_l)

    def compute_branch(self, pos_reshaped, inv_freq):
        freqs = pos_reshaped * inv_freq.view(1, 1, -1)
        c, s = torch.cos(freqs), torch.sin(freqs)
        t = pos_reshaped.shape[1]
        c_4d, s_4d = c.view(1, t, 1, HALF_DIM), s.view(1, t, 1, HALF_DIM)
        s_v_neg = s_4d * -1.0
        cos_out = torch.cat([c_4d, c_4d], dim=-1).view(1, t, 1, HEAD_DIM)
        sin_out = torch.cat([s_v_neg, s_4d], dim=-1).view(1, t, 1, HEAD_DIM)
        return cos_out, sin_out

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


def main():
    output_dir = "gemma3_quant_test/output"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_rope_fp32.tflite")
    final_path = os.path.join(output_dir, "gemma3_1b_rope_a16_GOLD.tflite")

    rope_mod = Gemma3RoPEOfficial().eval()
    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature(
        "prefill", rope_mod, sample_args=(torch.zeros(128, dtype=torch.int32),)
    )
    conv.add_signature(
        "decode", rope_mod, sample_args=(torch.zeros(1, dtype=torch.int32),)
    )
    conv.convert().export(tflite_path)

    qt = quantizer.Quantizer(float_model=tflite_path)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, symmetric=True, dtype=qtyping.TensorDataType.INT
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        skip_checks=True,
        explicit_dequantize=False,
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=qtyping.TFLOperationName.CONCATENATION,
        op_config=op_config,
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.OUTPUT, op_config=op_config
    )

    calib_data = {
        "prefill": [{"args_0": np.arange(128, dtype=np.int32)}],
        "decode": [{"args_0": np.array([1], dtype=np.int32)}],
    }
    res = qt.calibrate(calib_data)

    sin_scale = 0.000030518509447574615
    cos_scale = 0.00003051851308555342
    sin_max, cos_max = sin_scale * 32767, cos_scale * 32767

    for sig_res in res.values():
        if hasattr(sig_res, "tensor_quantization_stats"):
            for t_name, stats in sig_res.tensor_quantization_stats.items():
                target_max = cos_max if "cos" in t_name else sin_max
                try:
                    stats.min.fill(-target_max)
                    stats.max.fill(target_max)
                except:
                    pass

    quant_result = qt.quantize(res)
    quant_result.export_model(final_path, overwrite=True)

    import ai_edge_litert.interpreter as li_interp

    interp = li_interp.Interpreter(final_path)
    print("\n--- [GOLD STANDARD] FINAL SIGNATURE VERIFICATION ---")
    all_tensors = interp.get_tensor_details()

    for sig_def in interp._get_full_signature_list():
        print(f"\n[Signature: {sig_def['name']}]")
        for name, idx in sig_def["outputs"].items():
            t = all_tensors[idx]
            q = t.get("quantization_parameters", {})
            scale = q.get("scales", [0])[0]
            print(
                f"  Output: {name:<20} | tensor[{idx:<2}] | Type: {t['dtype']} | Scale: {scale:.12f} | Shape: {t['shape']}"
            )


if __name__ == "__main__":
    main()
