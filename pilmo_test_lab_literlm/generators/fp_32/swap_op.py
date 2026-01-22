import os
import sys
import numpy as np
from ai_edge_litert import schema_py_generated as schema
import flatbuffers


def swap_embedding_to_gather(input_path, output_path):
    print(f"Reading model from {input_path}...")
    with open(input_path, "rb") as f:
        model_content = bytearray(f.read())

    # We use a mutable model structure if possible, but flatbuffers schema is immutable view.
    # We need to rebuild it or use a raw byte modification if the indices are known.
    # Gemma 1B is 1.2GB, so raw byte modification is much safer/faster.

    # However, to be correct, we should use the schema to find indices.
    model_obj = schema.Model.GetRootAsModel(model_content, 0)
    subgraph = model_obj.Subgraphs(0)

    # 1. Find EMBEDDING_LOOKUP op
    found_op = None
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode_idx = op.OpcodeIndex()
        opcode = model_obj.OperatorCodes(opcode_idx).BuiltinCode()
        if opcode == schema.BuiltinOperator.EMBEDDING_LOOKUP:
            print(f"Found EMBEDDING_LOOKUP op at index {i}")
            found_op = op
            break

    if not found_op:
        print("ERROR: EMBEDDING_LOOKUP not found!")
        return

    # 2. Check if GATHER opcode already exists, otherwise add it
    gather_opcode_idx = -1
    for i in range(model_obj.OperatorCodesLength()):
        if model_obj.OperatorCodes(i).BuiltinCode() == schema.BuiltinOperator.GATHER:
            gather_opcode_idx = i
            break

    print(
        f"Self-Correction: Flatbuffer modification is tedious. I will write a script that creates a GATHER model from scratch OR use a simpler wrapper."
    )


if __name__ == "__main__":
    pass
