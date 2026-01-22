import tensorflow as tf
import os


def analyze_model(path):
    print(f"\n{'=' * 20} Analyzing: {path} {'=' * 20}")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        # Load without allocating to avoid custom op prep failure
        interpreter = tf.lite.Interpreter(model_path=path)

        tensor_details = interpreter.get_tensor_details()

        print(f"Total Tensors: {len(tensor_details)}")

        # Heuristic to find I/O tensors since get_signature_list is being difficult
        # We look for tensors that are likely I/O
        # For Main: embeddings, mask, pos_emb, kv_cache, logits
        # For Embedder: tokens, embeddings
        # For Aux: tokens, input_pos, rope, mask, cache_update

        keywords = [
            "token",
            "embedding",
            "mask",
            "pos",
            "cache",
            "logits",
            "input",
            "output",
        ]

        found_tensors = []
        for t in tensor_details:
            name_lower = t["name"].lower()
            if any(k in name_lower for k in keywords):
                found_tensors.append(t)

        # Sort by name for readability
        found_tensors.sort(key=lambda x: x["name"])

        print("\nRelevant Tensors found:")
        for t in found_tensors:
            print(
                f"  - {t['name']:40s}: Index {t['index']:3d}, Shape {str(t['shape']):15s}, Dtype {t['dtype']}"
            )

        ops = interpreter._get_ops_details()
        op_counts = {}
        for op in ops:
            name = op.get("op_name", "Unknown")
            op_counts[name] = op_counts.get(name, 0) + 1

        print(f"\nOperator Summary (Total: {len(ops)}):")
        for op_name, count in sorted(
            op_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  - {op_name:25s}: {count}")

    except Exception as e:
        print(f"Error analyzing {path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_io_types.py <model1.tflite> <model2.tflite> ...")
        sys.exit(1)

    for f in sys.argv[1:]:
        analyze_model(f)
