import os
import torch
import torch.nn as nn
import numpy as np
import ai_edge_torch
from ai_edge_quantizer import quantizer
from ai_edge_quantizer import qtyping

VOCAB_SIZE = 262144
EMBED_DIM = 1152


class GatherEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(VOCAB_SIZE, EMBED_DIM))

    def forward(self, input):
        return self.weight[input]


def test_embed_quant():
    model = GatherEmbed().eval()
    tflite_path = "test_embed_fp32.tflite"
    quant_path = "test_embed_w4.tflite"

    edge_model = ai_edge_torch.convert(model, (torch.zeros((1, 1), dtype=torch.int32),))
    edge_model.export(tflite_path)

    qt = quantizer.Quantizer(float_model=tflite_path)
    op_config = qtyping.OpQuantizationConfig(
        activation_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16,
            symmetric=True,
            granularity=qtyping.QuantGranularity.TENSORWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=4,
            symmetric=True,
            granularity=qtyping.QuantGranularity.CHANNELWISE,
            dtype=qtyping.TensorDataType.INT,
        ),
        compute_precision=qtyping.ComputePrecision.INTEGER,
        explicit_dequantize=False,
    )
    qt.update_quantization_recipe(
        regex=".*", operation_name=qtyping.TFLOperationName.GATHER, op_config=op_config
    )

    calib = {"main": [{"args_0": np.zeros((1, 1), dtype=np.int32)}]}
    res = qt.calibrate(calib)
    quant = qt.quantize(res)
    quant.export_model(quant_path, overwrite=True)

    print(f"FP32 Size: {os.path.getsize(tflite_path) / 1e6:.2f} MB")
    print(f"W4 Size: {os.path.getsize(quant_path) / 1e6:.2f} MB")


if __name__ == "__main__":
    test_embed_quant()
