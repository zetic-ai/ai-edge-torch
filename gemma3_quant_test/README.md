# Gemma3 Quantization Test Lab

INT8 Static Quantization 테스트를 위한 디렉토리입니다.

## 파일 구조

```
gemma3_quant_test/
├── README.md                      # 이 파일
├── gemma3_main_int8_test.py       # Gemma3-1B Main 모델 INT8 양자화 테스트
└── output/                        # 생성된 모델 파일들
    ├── gemma3_1b_main_fp32.tflite      # FP32 원본 모델
    ├── gemma3_1b_main_int8.tflite      # INT8 양자화 모델
    └── gemma3_1b_main_int8_aot.tflite  # QNN AOT 컴파일된 최종 모델
```

## 실행 방법

```bash
cd /home/pilmo/workspace/ai-edge-torch
python gemma3_quant_test/gemma3_main_int8_test.py
```

## 양자화 설정

- **Activation**: INT8 Asymmetric
- **Weight**: INT8 Symmetric
- **Target**: Qualcomm QNN (SM8750)
- **Signatures**: decode, prefill_128

## 프로세스 단계

1. **FP32 Export**: PyTorch → TFLite FP32
2. **Quantization Config**: Static INT8 설정
3. **Calibration**: 샘플 데이터로 양자화 범위 계산
4. **Quantization**: INT8 모델 생성
5. **AOT Compilation**: Qualcomm QNN 백엔드로 컴파일

## 참고

- `fusion_test_lab/toy_gather_fusion_test.py` 패턴을 따름
- ai_edge_quantizer의 Static Quantization API 사용
- LiteRT kernel 제약사항:
  - INT16 activation: Symmetric only
  - INT8 activation: Asymmetric
  - Weight: Always Symmetric (INT8/INT16 모두)
