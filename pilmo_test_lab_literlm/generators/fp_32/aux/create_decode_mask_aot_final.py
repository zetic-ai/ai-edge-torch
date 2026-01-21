import os
import torch
import torch.nn as nn
import ai_edge_torch
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

# Model Configuration (Gemma3-1B)
MASK_LEN_DECODE = 1281
MASK_LEN_PREFILL = 1408
SLIDING_WINDOW_SIZE = 512


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
            curr_kv_len = MASK_LEN_DECODE
            eff_q_pos = time_step.view(1)  # [1]
        else:
            curr_kv_len = MASK_LEN_PREFILL
            # For prefill, eff_q_pos is [seq_len]
            eff_q_pos = time_step + torch.arange(seq_len, device=input_tokens.device)

        kv_idx = torch.arange(curr_kv_len, device=input_tokens.device).view(1, 1, 1, -1)

        # 2. Mask Logic (Broadcast to [1, 1, T, KV_LEN])
        is_causal = kv_idx <= eff_q_pos.view(1, 1, -1, 1)
        is_window = kv_idx > (eff_q_pos.view(1, 1, -1, 1) - SLIDING_WINDOW_SIZE)

        mg = torch.where(is_causal, 0.0, self.mask_val) * gate
        ml = torch.where(is_causal & is_window, 0.0, self.mask_val) * gate

        # ONLY 2 OUTPUTS AS PER OFFICIAL SPEC
        return {"mask_global": mg, "mask_local": ml}


def main():
    output_dir = "./pilmo_test_lab_literlm/bin"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "gemma3_1b_aux.tflite")
    aot_tflite_path = os.path.join(output_dir, "gemma3_1b_aux_aot.tflite")

    decode_aux = Gemma3MaskGenerator(mode="decode").eval()
    prefill_aux = Gemma3MaskGenerator(mode="prefill").eval()

    print("Converting Gemma3 Auxiliary Model (Reset to 2 Outputs)...")
    conv = ai_edge_torch._convert.converter.Converter()

    # Signature 1: Prefill
    conv.add_signature(
        "prefill_128",
        prefill_aux,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.zeros((1, 128), dtype=torch.int32),
        },
    )

    # Signature 2: Decode
    conv.add_signature(
        "decode",
        decode_aux,
        sample_kwargs={
            "time_step": torch.tensor(0, dtype=torch.int32),
            "input_tokens": torch.tensor([[0]], dtype=torch.int32),
        },
    )

    edge_model = conv.convert()
    edge_model.export(tflite_path)

    print("AOT Compiling (Fusing to single DISPATCH_OP)...")
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
            print(
                f"SUCCESS: Official 2-output Aux AOT Model saved at {aot_tflite_path}"
            )
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
