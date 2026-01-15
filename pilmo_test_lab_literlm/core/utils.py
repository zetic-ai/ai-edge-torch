import os
from huggingface_hub import snapshot_download
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache


def download_from_hf(repo_id):
    print(f"\n[1/3] Downloading from Hugging Face: {repo_id}")
    return snapshot_download(repo_id)


def get_base_export_config():
    config = ExportConfig()
    config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    config.mask_as_input = True
    return config
