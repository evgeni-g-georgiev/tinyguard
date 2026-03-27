"""Inspect YAMNet TFLite model to find internal tensor details."""

import numpy as np
from ai_edge_litert.interpreter import Interpreter

model_path = "models/yamnet/yamnet.tflite"
interp = Interpreter(model_path=model_path)
interp.allocate_tensors()

print("=== INPUT DETAILS ===")
for inp in interp.get_input_details():
    print(f"  index={inp['index']}  name={inp['name']}  shape={inp['shape']}  dtype={inp['dtype']}")

print("\n=== OUTPUT DETAILS ===")
for out in interp.get_output_details():
    print(f"  index={out['index']}  name={out['name']}  shape={out['shape']}  dtype={out['dtype']}")

print("\n=== ALL TENSORS (looking for 1024D) ===")
for i, t in enumerate(interp.get_tensor_details()):
    shape = t['shape']
    # Show tensors that might be embeddings (1024D or large 1D/2D)
    if len(shape) >= 1 and any(d == 1024 for d in shape):
        print(f"  index={t['index']:4d}  name={t['name']:<60s}  shape={shape}")

print("\n=== ALL TENSORS WITH >256 FEATURES (broader search) ===")
for t in interp.get_tensor_details():
    shape = t['shape']
    if len(shape) >= 1 and any(d > 256 for d in shape):
        print(f"  index={t['index']:4d}  name={t['name']:<60s}  shape={shape}")
