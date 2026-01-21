import os
import torch
import torch.nn as nn
import ai_edge_torch
import ai_edge_torch.generative.custom_ops.dynamic_update_slice as dus_utils
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

HEAD_DIM = 256


class Gemma3CacheUpdate(nn.Module):
    def __init__(self, layer_indices=range(26), kv_cache_max_len=1280):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.kv_cache_max_len = kv_cache_max_len

    def forward(self, input_pos: torch.Tensor, **kwargs):
        """
        Gemma3 Cache Update: Exact Full Graph Reproduction.
        - Covers all layers from 0 to 25.
        - input_pos is expected as a tensor of shape [1].
        - Replicates Cast -> Concat -> Cast pattern for DynamicUpdateSlice indices.
        """
        results = {}

        # 1. Cast input_pos [1] to float32
        pos_f = input_pos.float()  # [1]
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


def main():
    # Full layers from 0 up to 25
    layer_indices = range(26)
    kv_cache_max_len = 1280

    output_dir = "./cache_update_aot_final"
    os.makedirs(output_dir, exist_ok=True)
    tflite_path = os.path.join(output_dir, "cache_update_full_system.tflite")

    model = Gemma3CacheUpdate(
        layer_indices=layer_indices, kv_cache_max_len=kv_cache_max_len
    )
    model.eval()

    # Matching inputs: input_pos set to tensor of shape [1]
    sample_kwargs = {
        "input_pos": torch.tensor([0], dtype=torch.int32),
    }
    for i in layer_indices:
        # K cache matches user's log: [1, 1, 1280, 256]
        sample_kwargs[f"kv_cache_k_{i}"] = torch.zeros(
            (1, 1, kv_cache_max_len, HEAD_DIM), dtype=torch.float32
        )
        sample_kwargs[f"kv_slice_k_{i}"] = torch.zeros(
            (1, 1, 1, HEAD_DIM), dtype=torch.float32
        )

        # V cache matches user's log: [1, 1, 256, 1280]
        sample_kwargs[f"kv_cache_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, kv_cache_max_len), dtype=torch.float32
        )
        sample_kwargs[f"kv_slice_v_{i}"] = torch.zeros(
            (1, 1, HEAD_DIM, 1), dtype=torch.float32
        )

    print(f"Exporting CacheUpdate with FULL layers (0-25) and input_pos [1]...")
    conv = ai_edge_torch._convert.converter.Converter()
    conv.add_signature("decode_cache_update", model, sample_kwargs=sample_kwargs)
    edge_model = conv.convert()
    edge_model.export(tflite_path)
    print(f"Full System TFLite saved at {tflite_path}")

    # Optional: AOT Compile for SM8750
    aot_tflite_path = tflite_path.replace(".tflite", "_aot.tflite")
    print("Starting AOT compilation...")
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
            print(f"SUCCESS: AOT Model saved at {aot_tflite_path}")
    except Exception as e:
        print(f"AOT Error: {e}")


if __name__ == "__main__":
    main()
