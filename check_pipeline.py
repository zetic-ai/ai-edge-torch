import tensorflow as tf
import os


def get_sig_details(interpreter, sig_name):
    try:
        runner = interpreter.get_signature_runner(sig_name)
        inputs = {}
        for name, detail in runner._inputs.items():
            inputs[name] = {
                "shape": list(detail["shape"]),
                "dtype": str(detail["dtype"]),
            }
        outputs = {}
        for name, detail in runner._outputs.items():
            outputs[name] = {
                "shape": list(detail["shape"]),
                "dtype": str(detail["dtype"]),
            }
        return inputs, outputs
    except Exception as e:
        return f"Error: {e}", None


def run_full_analysis():
    models = {
        "Embedder": "pilmo_test_lab_literlm/bin/embedder_int8/gemma3_1b_embedder_int8.tflite",
        "Aux": "pilmo_test_lab_literlm/bin/aux_int8/gemma3_1b_aux_int8.tflite",
        "Main": "pilmo_test_lab_literlm/bin/main_int8/gemma3_1b_main_int8_aot.tflite",
    }

    results = {}
    for key, path in models.items():
        if not os.path.exists(path):
            results[key] = "File missing"
            continue

        interpreter = tf.lite.Interpreter(model_path=path)
        sigs = interpreter.get_signature_list()
        results[key] = {}
        for sig_name in sigs.keys():
            inputs, outputs = get_sig_details(interpreter, sig_name)
            results[key][sig_name] = {"inputs": inputs, "outputs": outputs}

    # Print analysis Report
    print("\n" + "=" * 80)
    print(" PIPELINE I/O MATCHING REPORT ")
    print("=" * 80)

    # 1. Embedder -> Main
    print("\n[Step 1] Embedder -> Main")
    emb_out = (
        results["Embedder"]["decode_embedder"]["outputs"]
        if "Embedder" in results and "decode_embedder" in results["Embedder"]
        else None
    )
    main_in = (
        results["Main"]["decode"]["inputs"]
        if "Main" in results and "decode" in results["Main"]
        else None
    )

    if emb_out and main_in:
        e_out = emb_out.get("embeddings")
        m_in = main_in.get("embeddings")
        print(f"  Embedder Out ('embeddings'): {e_out}")
        print(f"  Main In ('embeddings')    : {m_in}")
        match = "OK" if e_out == m_in else "MISMATCH"
        print(f"  >> Match Status: {match}")
    else:
        print("  >> Could not compare (Signatures missing)")

    # 2. Aux (Rope/Mask) -> Main
    print("\n[Step 2] Aux (Prep) -> Main")
    aux_rope_out = (
        results["Aux"]["decode_rope"]["outputs"]
        if "Aux" in results and "decode_rope" in results["Aux"]
        else None
    )
    aux_mask_out = (
        results["Aux"]["decode_mask"]["outputs"]
        if "Aux" in results and "decode_mask" in results["Aux"]
        else None
    )

    checks = [
        ("pos_emb_cos", aux_rope_out, main_in),
        ("pos_emb_sin", aux_rope_out, main_in),
        ("mask_global", aux_mask_out, main_in),
        ("mask_local", aux_mask_out, main_in),
    ]

    for name, aux_out_dict, main_in_dict in checks:
        if aux_out_dict and main_in_dict:
            a_out = aux_out_dict.get(name)
            m_in = main_in_dict.get(name)
            print(f"  Aux Out ('{name}'): {a_out}")
            print(f"  Main In ('{name}'): {m_in}")
            match = "OK" if a_out == m_in else "MISMATCH"
            print(f"  >> Match Status: {match}")
        else:
            print(f"  >> {name}: Aux or Main signatures/dict missing")

    # 3. Main -> Cache Update
    print("\n[Step 3] Main -> Cache Update")
    main_out = (
        results["Main"]["decode"]["outputs"]
        if "Main" in results and "decode" in results["Main"]
        else None
    )
    cache_in = (
        results["Aux"]["decode_cache_update"]["inputs"]
        if "Aux" in results and "decode_cache_update" in results["Aux"]
        else None
    )

    if main_out and cache_in:
        m_out = main_out.get("kv_slice_k_0")
        c_in = cache_in.get("kv_slice_k_0")
        print(f"  Main Out ('kv_slice_k_0'): {m_out}")
        print(f"  Cache In ('kv_slice_k_0'): {c_in}")
        match = "OK" if m_out == c_in else "MISMATCH"
        print(f"  >> Match Status: {match}")
    else:
        print("  >> Could not compare Main outputs with Cache Update inputs")


run_full_analysis()
