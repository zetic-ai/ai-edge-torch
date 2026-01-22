class Gemma3Config:
    # Model architecture constants
    NUM_LAYERS = 26
    HEAD_DIM = 256
    EMBED_DIM = 1152
    VOCAB_SIZE = 262144

    # KV Cache constants
    KV_CACHE_LEN = 1280
    SLIDING_WINDOW = 512

    # Mask constants
    DECODE_MASK_LEN = 1281
    PREFILL_T = 128
    PREFILL_MASK_LEN = 1408  # KV_CACHE_LEN + PREFILL_T (1280 + 128)

    # Paths
    OUTPUT_BIN_DIR = "./pilmo_test_lab_literlm/bin"
    MAIN_BIN_DIR = "./pilmo_test_lab_literlm/bin/main"
