import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema
import sys

# Add path to schema if needed
sys.path.append("/home/pilmo/miniconda3/envs/litertlm/lib/python3.11/site-packages/tensorflow/lite/python")
try:
    import schema_py_generated as schema
except ImportError:
    pass

def check(path):
    with open(path, 'rb') as f:
        buf = f.read()
    model = schema.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)
    
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode()
        shape = [t.Shape(k) for k in range(t.ShapeLength())]
        if 256 in shape and (128 in shape or 1 in shape):
             # Only print if it's not a weight/constant (buffer 0 is usually not constant, but let's check)
             if t.Buffer() > 0: continue 
             print(f"  Tensor[{i}]: {name}, Shape {shape}")

check('gemma3_quant_test/output/w8_pilmo_optimized_main_w8a16_aot.tflite')
