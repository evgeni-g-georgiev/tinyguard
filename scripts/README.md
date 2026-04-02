# scripts/

Standalone utilities that don't belong to a specific pipeline stage.

## Files

| File | Description |
|---|---|
| `memory_audit.py` | Audits the flash and SRAM footprint of the full on-device pipeline |

## memory_audit.py

Computes the memory budget for both on-device phases (f_s training and inference) without requiring any data or a trained model. Optionally accepts a real `.tflite` file for an exact flash size check.

**Input:** none (uses model code only); optionally `--tflite <path>`

**Output:** printed report covering:
- Layer-by-layer flash breakdown (INT8 weights + INT32 biases + per-channel quant scales)
- Peak SRAM activation trace through the AcousticEncoder
- Total SRAM budget for inference and training phases

```bash
python scripts/memory_audit.py
python scripts/memory_audit.py --tflite distillation/outputs/export/acoustic_encoder_int8.tflite
```
