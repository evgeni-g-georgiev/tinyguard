# preprocessing/

Prepares cached features for distillation and fixes the MIMII train/test split. Run once; all outputs are reused by later stages.

## Scripts

| Script | What it does |
|---|---|
| `extract_embeddings.py` | Runs YAMNet over every FSD50K clip, caches 1024D embeddings, fits PCA(32D) |
| `compute_mels.py` | Computes log-mel spectrograms for every FSD50K clip, caches as numpy array |
| `split_mimii.py` | Assigns MIMII clips to train/test sets; saves a fixed manifest so all stages use identical splits |

Run them in order:

```bash
python preprocessing/extract_embeddings.py   # ~15–20 min
python preprocessing/compute_mels.py         # ~30 min, ~1.5 GB output
python preprocessing/split_mimii.py          # fast; requires data/mimii/ to be populated
```

Each script is idempotent — if its outputs already exist it prints a message and exits early.

## Inputs

| Source | Required by |
|---|---|
| `data/fsd50k/FSD50K.eval_audio/` | `extract_embeddings.py`, `compute_mels.py` |
| `data/yamnet/yamnet.tflite` | `extract_embeddings.py` |
| `data/mimii/` | `split_mimii.py` |

## Outputs

| File | Shape | Used by |
|---|---|---|
| `preprocessing/outputs/fsd50k_cache/eval_embeddings.npy` | (N, 1024) float32 | `distillation/train.py` |
| `preprocessing/outputs/pca/pca_components.npy` | (32, 1024) float32 | `distillation/train.py` |
| `preprocessing/outputs/pca/pca_mean.npy` | (1024,) float32 | `distillation/train.py` |
| `preprocessing/outputs/fsd50k_cache/eval_mels.npy` | (N, 1, 64, 61) float32 | `distillation/train.py` |
| `preprocessing/outputs/mimii_splits/splits.json` | JSON manifest | `separator/train.py`, `inference/run.py` |

## Notes

- PCA is fitted on FSD50K only. Fitting on MIMII would be data leakage.
- `splits.json` stores paths relative to `data/mimii/` and uses a fixed seed (42), so every run produces identical splits. Regenerating it does not change clip assignments.
- Frame index `i` in `eval_mels.npy` corresponds exactly to frame `i` in `eval_embeddings.npy` — the two scripts must process files in the same sorted order, which they do.
