import os
import tempfile
from huggingface_hub import snapshot_download
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache

# AOT specific imports (these will fail if nightly packages are not installed)
try:
    from ai_edge_litert.aot import aot_compile as aot_lib
    from ai_edge_litert.aot.core import types as litert_types
    from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target

    HAS_AOT = True
except ImportError:
    HAS_AOT = False


def download_from_hf(repo_id):
    print(f"\n[1/3] Downloading from Hugging Face: {repo_id}")
    return snapshot_download(repo_id)


def get_base_export_config():
    config = ExportConfig()
    config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    config.mask_as_input = True
    return config


def compile_for_qnn(tflite_path, output_path, soc_model_name="SM8750"):
    if not HAS_AOT:
        raise RuntimeError("ai-edge-litert-nightly is required for AOT compilation.")

    print(f"\n[AOT] Compiling for QNN ({soc_model_name})...")

    with open(tflite_path, "rb") as f:
        tflite_model_bytes = f.read()

    litert_model = litert_types.Model.create_from_bytes(tflite_model_bytes)

    # Convert string to SocModel enum
    target_enum = getattr(
        qnn_target.SocModel, soc_model_name.upper(), qnn_target.SocModel.SM8750
    )
    target = qnn_target.Target(soc_model=target_enum)
    config = [litert_types.CompilationConfig(target=target)]

    # We can try to catch and show more errors
    try:
        result = aot_lib.aot_compile(litert_model, config=config)
    except Exception as e:
        print(f"[AOT] Critical error during aot_compile: {e}")
        raise

    # compilation_result contains (backend, model) tuples
    if not result.models_with_backend:
        print(
            f"[AOT] Warning: Compilation result is empty. This usually means the QNN SDK is missing or model is incompatible."
        )
        raise RuntimeError("AOT compilation yielded no results.")

    # Usually the first one is our target
    backend, compiled_model = result.models_with_backend[0]
    compiled_model_bytes = compiled_model.model_bytes

    with open(output_path, "wb") as f:
        f.write(compiled_model_bytes)

    print(f"[AOT] SUCCESS: Compiled model saved to {output_path}")
    return output_path
