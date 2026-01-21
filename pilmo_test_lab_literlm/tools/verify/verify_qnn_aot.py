import os
import sys
from ai_edge_litert.aot import aot_compile as aot_lib
from ai_edge_litert.aot.core import types as litert_types
from ai_edge_litert.aot.vendors.qualcomm import target as qnn_target


def verify_whisper_aot():
    input_tflite = "/mnt/disks/zeticai_database/models/model_zoo/test-litert-convert/original_tflite/whisper_base_encoder.tflite"

    os.makedirs("./output", exist_ok=True)

    print(f"Loading model: {input_tflite}")
    with open(input_tflite, "rb") as f:
        tflite_model_bytes = f.read()

    litert_model = litert_types.Model.create_from_bytes(tflite_model_bytes)

    models_to_test = [qnn_target.SocModel.SM8750]

    for model_type in models_to_test:
        print(f"\n--- Testing Target: {model_type} ---")
        output_tflite = f"./output/whisper_base_encoder_{model_type}_qnn.tflite"
        target = qnn_target.Target(soc_model=model_type)
        config = [litert_types.CompilationConfig(target=target)]

        try:
            print("Starting AOT compilation...")
            result = aot_lib.aot_compile(litert_model, config=config)
            print(result)
            if result.models_with_backend:
                backend, compiled_model = result.models_with_backend[0]
                compiled_model_bytes = compiled_model.model_bytes

                with open(output_tflite, "wb") as f:
                    f.write(compiled_model_bytes)

                print(f"SUCCESS for {model_type}!")
                print(f"Saved to {output_tflite}")
                print(f"Backend: {backend.id()}")
            else:
                print(f"FAILED for {model_type}: No results returned from aot_compile.")
        except Exception as e:
            print(f"EXCEPTION for {model_type}: {e}")


if __name__ == "__main__":
    verify_whisper_aot()
