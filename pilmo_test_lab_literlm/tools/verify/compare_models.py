import tensorflow as tf
import os


def summarize_signature(path, signature_key):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    print(f"--- Analyzing Signature: {signature_key} in {os.path.basename(path)} ---")
    interpreter = tf.lite.Interpreter(model_path=path)

    try:
        signature_runner = interpreter.get_signature_runner(signature_key)
        # We can't easily get the OP flow just from the runner,
        # but we can look at the subgraphs.

        # In TFLite, signatures map to subgraphs.
        # Let's find which subgraph corresponds to this signature.
        details = interpreter._get_full_details()
        # This is tricky as 'signatures' in details might be available in some TF versions.
        # Alternatively, let's look at all subgraphs.

        print(f"Total Subgraphs: {len(details['subgraphs'])}")

        for sg_idx, sg in enumerate(details["subgraphs"]):
            print(f"\nSubgraph {sg_idx}:")
            # Print signature inputs/outputs if they match

            for op_idx, op in enumerate(sg["operators"]):
                op_code_idx = op["opcode_index"]
                op_code = details["operator_codes"][op_code_idx]

                # Resolve Op Name
                builtin_code = op_code["builtin_code"]
                custom_code = op_code["custom_code"]

                if custom_code:
                    op_name = f"CUSTOM ({custom_code})"
                else:
                    # In newer TF we'd use tf.lite.BuiltinOperator(builtin_code).name
                    # Here we just use a placeholder or common ones
                    op_name = f"BUILTIN ({builtin_code})"
                    # Map some common ones manually for clarity
                    mapping = {
                        0: "ADD",
                        1: "AVERAGE_POOL_2D",
                        2: "CONCATENATION",
                        3: "CONV_2D",
                        22: "RESHAPE",
                        25: "SOFTMAX",
                        32: "CUSTOM",
                        55: "CAST",
                        95: "FILL",
                        133: "DYNAMIC_UPDATE_SLICE",
                        49: "PACK",
                    }
                    if builtin_code in mapping:
                        op_name = mapping[builtin_code]

                print(
                    f"  [{op_idx:2d}] OP: {op_name:25s} (In:{op['inputs']}, Out:{op['outputs']})"
                )

            # Print Tensors in this subgraph
            print(f"  Tensors in Subgraph {sg_idx}:")
            for t_idx in range(len(sg["tensors"])):
                t = sg["tensors"][t_idx]
                if t_idx < 50:  # Limit output
                    print(f"    Tensor[{t_idx:2d}]: {t['name']:30s} Shape:{t['shape']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    succeed_model_path = "/home/pilmo/workspace/ai-edge-torch/pilmo_test_lab_literlm/succeed_model/real_aux.tflite"
    # We'll just look at Subgraph 0 if signature mapping is hard, or list all.
    summarize_signature(succeed_model_path, "decode_cache_update")
