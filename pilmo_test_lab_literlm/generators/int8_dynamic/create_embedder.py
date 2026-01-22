import os
import sys
import torch
import ai_edge_torch
from ai_edge_torch.generative.quantize import quant_recipes

# Add parent directory to sys.path to access common files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_config import Gemma3Config
from common_models import Gemma3Embedder, Gemma3EmbedderSignatureWrapper
from export_utils import export_and_compile


def main():
    output_dir = os.path.join(Gemma3Config.OUTPUT_BIN_DIR, "embedder_int8")
    tflite_path = os.path.join(output_dir, "gemma3_1b_embedder_int8.tflite")
    model = Gemma3EmbedderSignatureWrapper(Gemma3Embedder()).eval()

    conv = ai_edge_torch._convert.converter.Converter()

    # 1. decode_embedder Signature
    conv.add_signature(
        "decode_embedder",
        model,
        sample_kwargs={"token_ids": torch.zeros((1, 1), dtype=torch.int32)},
    )

    # 2. prefill_embedder_128 Signature
    conv.add_signature(
        "prefill_embedder_128",
        model,
        sample_kwargs={
            "token_ids": torch.zeros((1, Gemma3Config.PREFILL_T), dtype=torch.int32)
        },
    )

    # Apply INT8 Weighted-Only Dynamic Quantization
    quant_config = quant_recipes.full_dynamic_recipe()

    export_and_compile(conv, tflite_path, quant_config=quant_config, run_aot=False)


if __name__ == "__main__":
    main()
