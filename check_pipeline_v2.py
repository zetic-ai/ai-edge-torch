import tensorflow as tf
import os


def get_model_io(path):
    if not os.path.exists(path):
        return None
    interpreter = tf.lite.Interpreter(model_path=path)
    sigs = interpreter.get_signature_list()
    tensor_details = {t["index"]: t for t in interpreter.get_tensor_details()}

    report = {}
    for sig_name, sig_def in sigs.items():
        report[sig_name] = {"inputs": {}, "outputs": {}}
        # Some TF versions return a list of inputs, some a dict
        inputs_raw = sig_def.get("inputs", {})
        outputs_raw = sig_def.get("outputs", {})

        if isinstance(inputs_raw, dict):
            for name, idx in inputs_raw.items():
                t = tensor_details.get(idx)
                if t:
                    report[sig_name]["inputs"][name] = {
                        "shape": list(t["shape"]),
                        "dtype": str(t["dtype"]),
                    }

        if isinstance(outputs_raw, dict):
            for name, idx in outputs_raw.items():
                t = tensor_details.get(idx)
                if t:
                    report[sig_name]["outputs"][name] = {
                        "shape": list(t["shape"]),
                        "dtype": str(t["dtype"]),
                    }

    return report


def compare():
    emb = get_model_io(
        "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8.tflite"
    )
    aux = get_model_io("pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8.tflite")
    main = get_model_io(
        "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite"
    )

    print("\n[V] VERIFYING MODEL PIPELINE CONSISTENCY\n")

    # Check 1: Embedder -> Main
    print("1. Embedder -> Main")
    e_out = emb["decode_embedder"]["outputs"].get("embeddings") if emb else None
    m_in = main["decode"]["inputs"].get("embeddings") if main else None
    print(f"   Embedder (decode_embedder) Out: {e_out}")
    print(f"   Main (decode) In              : {m_in}")

    # Check 2: Aux -> Main (Rope)
    print("\n2. Aux (Rope) -> Main")
    a_out = aux["decode_rope"]["outputs"].get("pos_emb_cos") if aux else None
    m_in = main["decode"]["inputs"].get("pos_emb_cos") if main else None
    print(f"   Aux (decode_rope) Out: {a_out}")
    print(f"   Main (decode) In     : {m_in}")

    # Check 3: Aux -> Main (Mask)
    print("\n3. Aux (Mask) -> Main")
    a_out = aux["decode_mask"]["outputs"].get("mask_global") if aux else None
    m_in = main["decode"]["inputs"].get("mask_global") if main else None
    print(f"   Aux (decode_mask) Out: {a_out}")
    print(f"   Main (decode) In     : {m_in}")

    # Check 4: Main -> Cache Update
    print("\n4. Main -> Cache Update")
    m_out = main["decode"]["outputs"].get("kv_slice_k_0") if main else None
    c_in = aux["decode_cache_update"]["inputs"].get("kv_slice_k_0") if aux else None
    print(f"   Main (decode) Out             : {m_out}")
    print(f"   Aux (decode_cache_update) In  : {c_in}")


compare()
