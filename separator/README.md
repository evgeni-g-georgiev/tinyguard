# separator/

Simulates the on-device 10-minute training window. For each of the 16 MIMII machines, loads 60 normal clips, extracts embeddings via the frozen AcousticEncoder, trains a Deep SVDD model (f_s), and saves the resulting artefact.

## Files

| File | Description |
|---|---|
| `separator.py` | `FsSeparator` model definition. Also contains `train_fs()` (SVDD training loop) and `score_clips()` (max-frame L2 scoring). |
| `train.py` | Per-machine training script. Reads clip paths from the splits manifest, runs the training loop, saves artefacts. |

## Usage

```bash
python separator/train.py
python separator/train.py --checkpoint distillation/outputs/student/acoustic_encoder.pt
```

## Inputs

| File | Description |
|---|---|
| `distillation/outputs/student/acoustic_encoder.pt` | Frozen AcousticEncoder weights |
| `preprocessing/outputs/mimii_splits/splits.json` | Clip manifest — provides `train_normal` paths per machine |
| `data/mimii/` | MIMII WAV files |

## Outputs

One `.pt` file per machine in `separator/outputs/separator/`:

```
separator/outputs/separator/
  fan_id_00.pt
  fan_id_02.pt
  ...
  valve_id_06.pt    (16 files total)
```

Each file contains:

```python
{
  "state_dict":  FsSeparator weights,
  "centroid":    (8,) float32 — fixed SVDD hypersphere centre,
  "threshold":   float — 95th percentile of training scores,
  "input_dim":   32,
  "hidden_dim":  32,
  "output_dim":  8,
  "n_params":    1312,
}
```

## Model: FsSeparator

```
Linear(32 → 32, bias=True) + ReLU
Linear(32 → 8,  bias=False)           ← no bias required by Deep SVDD
```

The final layer has no bias to prevent the trivial collapse solution (W=0, b=centroid). 1,312 parameters, ~5 KB at float32.

## SVDD training

Simulates the realistic on-device flow:
- During the 10-minute collection window, each audio frame is processed through the frozen AcousticEncoder and the 32D embedding stored (~75KB total — well within the 786KB SRAM budget). Raw audio is discarded after each frame.
- After collection, FsSeparator is trained on all stored embeddings for a fixed 50 epochs. At ~24–48 ms/epoch on Cortex-M33 @ 160 MHz (~2.4M MACs/epoch), 50 epochs takes ~1–2.5 seconds.

Training steps:
1. Forward pass all training embeddings → compute centroid `c = mean(f_s(x))`, then freeze it.
2. Minimise `L = mean(||f_s(x) − c||²)` via SGD (lr=0.01, weight decay=1e-4) for 50 fixed epochs.
3. Cosine LR decay over the 50 epochs for stable convergence.
4. Set threshold `τ = percentile(scores, 95)` on all training clips.
