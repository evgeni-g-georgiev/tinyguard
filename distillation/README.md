# distillation/

Trains the AcousticEncoder (f_c) — the frozen CNN that runs on the Arduino. Uses knowledge distillation: the student CNN learns to reproduce YAMNet's embeddings (compressed to 32D via PCA) via MSE loss on FSD50K audio.

## Files

| File | Description |
|---|---|
| `cnn.py` | `AcousticEncoder` — MobileNet V1-style depthwise-separable CNN. Input: (1, 64, 61) log-mel. Output: 32D embedding. ~554K params, ~562 KB INT8. |
| `train.py` | Training script. Loads FSD50K mel/embedding caches, projects targets via PCA, trains via AdamW + cosine LR schedule. |

## Usage

```bash
python distillation/train.py                            # 50 epochs, default settings
python distillation/train.py --epochs 100 --lr 1e-3    # custom run
```

## Inputs

| File | Description |
|---|---|
| `preprocessing/outputs/fsd50k_cache/eval_mels.npy` | Log-mel spectrograms (N, 1, 64, 61) |
| `preprocessing/outputs/fsd50k_cache/eval_embeddings.npy` | YAMNet embeddings (N, 1024) |
| `preprocessing/outputs/pca/pca_components.npy` | PCA projection matrix (32, 1024) |
| `preprocessing/outputs/pca/pca_mean.npy` | PCA mean vector (1024,) |

## Outputs

| File | Description |
|---|---|
| `distillation/outputs/student/acoustic_encoder.pt` | Best checkpoint (lowest val MSE). Contains `model_state_dict`, `epoch`, `val_loss`. |
| `distillation/outputs/student/training_curve.png` | Train / val MSE loss curves |

## Architecture

```
Input (1, 64, 61)
  → Stem Conv 3×3 stride 2
  → 7× DepthwiseSeparableBlock (stride 1 or 2)
  → AdaptiveAvgPool → Linear(512 → 32)
Output (32,)
```

MIMII data is never used here. The encoder learns a general audio feature space from FSD50K and generalises to factory machinery at evaluation time.
