import os
import torch
from huggingface_hub import snapshot_download
from ai_edge_torch.generative.examples.gemma3 import gemma3
from ai_edge_torch.generative.utilities import converter
from ai_edge_torch.generative.utilities.export_config import ExportConfig
from ai_edge_torch.generative.layers import kv_cache


def main():
    repo_id = "google/gemma-3-270m-it"
    output_dir = "./output"

    # 1. Hugging Face에서 체크포인트 다운로드
    print(f"Downloading weights from Hugging Face: {repo_id}...")
    checkpoint_dir = snapshot_download(repo_id)
    print(f"Weights downloaded to: {checkpoint_dir}")

    # 2. Gemma 3-270M PyTorch 모델 빌드
    # 이 함수는 내부적으로 checkpoint_dir의 safetensors를 로드합니다.
    print("Building PyTorch model...")
    pytorch_model = gemma3.build_model_270m(checkpoint_dir)

    # 3. 내보내기 설정 (KV Cache 최적화)
    export_config = ExportConfig()
    export_config.kvcache_layout = kv_cache.KV_LAYOUT_TRANSPOSED
    export_config.mask_as_input = True

    # 4. LiteRT-LM (.litertlm) 번들링용 메타데이터 설정
    # Gemma 3-it(Instruction Tuned) 모델의 표준 프롬프트 템플릿을 적용합니다.
    litertlm_config = {
        "tokenizer_model_path": os.path.join(checkpoint_dir, "tokenizer.model"),
        "start_token_id": 2,  # "<bos>"
        "stop_token_ids": [1, 106],  # ["<eos>", "<end_of_turn>"]
        "user_prompt_prefix": "<start_of_turn>user\n",
        "user_prompt_suffix": "<end_of_turn>\n",
        "model_prompt_prefix": "<start_of_turn>model\n",
        "model_prompt_suffix": "<end_of_turn>\n",
        "output_format": "litertlm",  # .litertlm 번들 생성을 지시
    }

    print("Starting conversion to .litertlm format...")
    os.makedirs(output_dir, exist_ok=True)

    # 5. 변환 실행
    try:
        # 이 함수가 내부적으로 TFLite 변환 후 .litertlm으로 패키징합니다.
        converter.convert_to_litert(
            pytorch_model,
            output_path=output_dir,
            output_name_prefix="gemma-3-270m",
            prefill_seq_len=1024,  # 모바일 기기 성능을 고려한 적절한 길이
            kv_cache_max_len=2048,
            quantize="dynamic_int8",  # 8-bit 양자화 적용
            export_config=export_config,
            **litertlm_config,
        )
        print("\n" + "=" * 50)
        print(
            f"SUCCESS: Result saved at {os.path.join(output_dir, 'gemma-3-270m.litertlm')}"
        )
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAILED] Conversion error: {e}")
        # .litertlm 빌더는 ai-edge-litert-nightly에 포함된 경우가 많습니다.
        if "LiteRT-LM builder" in str(e):
            print("\nAdvice: You need the nightly packages for the .litertlm builder.")
            print(
                "Try command: pip install ai-edge-torch-nightly ai-edge-litert-nightly"
            )


if __name__ == "__main__":
    main()
