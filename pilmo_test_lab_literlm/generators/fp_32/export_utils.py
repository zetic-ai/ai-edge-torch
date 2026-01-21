import os
import ai_edge_torch
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


def run_aot_compilation(tflite_path, aot_tflite_path, signature_names=None):
    """
    Runs QNN AOT compilation for the given TFLite model.
    If signature_names is None, compiles all signatures.
    """
    print(f"Starting AOT compilation for {os.path.basename(tflite_path)}...")
    try:
        with open(tflite_path, "rb") as f:
            tflite_bytes = f.read()

        litert_model = litert_types.Model.create_from_bytes(tflite_bytes)
        target = qnn_target.Target(soc_model=qnn_target.SocModel.SM8750)

        config = [
            litert_types.CompilationConfig(
                target=target, signature_names=signature_names
            )
        ]

        result = aot_lib.aot_compile(litert_model, config=config)
        if result.models_with_backend:
            with open(aot_tflite_path, "wb") as f:
                f.write(result.models_with_backend[0][1].model_bytes)
            print(f"SUCCESS: AOT Model saved at {aot_tflite_path}")
            if signature_names:
                print(f"AOT targets: {signature_names}")
            return True
    except Exception as e:
        print(f"AOT Error for {tflite_path}: {e}")
    return False


def export_and_compile(conv, output_path, aot_signatures=None):
    """
    Converts via Converter, exports TFLite, and optionally runs AOT.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    aot_path = output_path.replace(".tflite", "_aot.tflite")

    print(f"Converting and exporting to {output_path}...")
    edge_model = conv.convert()
    edge_model.export(output_path)

    if aot_signatures is not False:  # Pass None for all, False to skip
        run_aot_compilation(output_path, aot_path, signature_names=aot_signatures)
