# Gemma3 Quantization Test Lab

Gemma3-1B 모델의 정적 양자화 테스트를 위한 디렉토리입니다.

## 파일 구조

```
gemma3_quant_test/
├── README.md                      # 이 파일
├── gemma3_main_int8_test.py       # INT8 양자화 + AOT (성능 최적화)
├── gemma3_main_int16_test.py      # INT16 I/O (런타임 호환)
└── output/                        # 생성된 모델 파일들
    ├── gemma3_1b_main_fp32.tflite          # FP32 원본 모델
    ├── gemma3_1b_main_int8.tflite          # INT8 양자화 모델
    ├── gemma3_1b_main_int8_aot.tflite      # QNN AOT 컴파일 (추천!)
    └── gemma3_1b_main_int16_w8.tflite      # INT16 I/O + INT8 Weight
```

## 두 가지 버전

### 1. INT8 버전 (권장 - 성능 최적화)

```bash
python gemma3_quant_test/gemma3_main_int8_test.py
```

**특징:**
- ✅ Activation: INT8 (Asymmetric)
- ✅ Weight: INT8 (Symmetric)
- ✅ QNN AOT 컴파일 성공
- ✅ Qualcomm NPU 가속
- ⚠️ 입출력: INT8 (일부 런타임 제약 있음)

**생성 파일:**
- `gemma3_1b_main_int8.tflite` (972MB)
- `gemma3_1b_main_int8_aot.tflite` (1.9GB) - **실제 사용 권장**

### 2. INT16 버전 (런타임 호환)

```bash
python gemma3_quant_test/gemma3_main_int16_test.py
```

**특징:**
- ✅ Activation: INT16 (Symmetric)
- ✅ Weight: INT8 (Symmetric)
- ✅ 입출력: INT16 (런타임 제약 충족)
- ❌ QNN AOT 컴파일 불가 (QNN이 INT16 activation 미지원)
- ⚠️ CPU/GPU fallback

**생성 파일:**
- `gemma3_1b_main_int16_w8.tflite` (972MB)

## 프로세스 비교

| 단계 | INT8 버전 | INT16 버전 |
|------|----------|-----------|
| FP32 Export | ✅ | ✅ |
| Static Quantization | INT8 | INT16 (I/O) + INT8 (W) |
| Calibration | ✅ | ✅ |
| AOT Compilation | ✅ QNN HTP | ❌ 불가 |
| 최종 크기 | 1.9GB | 972MB |
| 추론 속도 | 🚀 빠름 (NPU) | 🐢 느림 (CPU) |

## 선택 가이드

**INT8 버전을 사용하세요:**
- 성능이 중요한 경우
- Qualcomm 디바이스에서 실행
- 런타임 제약이 없는 경우

**INT16 버전을 사용하세요:**
- 런타임이 INT16/FP32 I/O만 지원하는 경우
- CPU/GPU에서 실행
- AOT 컴파일이 불필요한 경우

## 참고사항

- `fusion_test_lab/toy_gather_fusion_test.py` 패턴 적용
- LiteRT kernel 제약:
  - INT16 activation: Symmetric only
  - INT8 activation: Asymmetric  
  - Weight: Always Symmetric (INT8/INT16 모두)
- QNN HTP 제약:
  - INT16 activation은 일부 연산에서 미지원
  - INT8 activation 권장