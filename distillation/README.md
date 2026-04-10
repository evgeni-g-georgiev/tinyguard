# distillation/

This folder trains the frozen acoustic embedder used by the CNN + SVDD pipeline. This folder owns the full distillation workflow:
1. extract YAMNet teacher embeddings from FSD50K,
2. fit the PCA projection used as the 32D training target,
3. compute the aligned log-mel spectrogram cache used as student input,
4. train the `AcousticEncoder` to reproduce the PCA-projected teacher targets.

## Folder Structure

```text
distillation/
├── README.md
├── __init__.py
├── extract_embeddings.py
├── compute_mels.py
├── cnn.py
├── training_pipeline.py
├── train.py
└── outputs/
    ├── fsd50k_cache/
    │   ├── eval_embeddings.npy
    │   └── eval_mels.npy
    ├── pca/
    │   ├── pca_components.npy
    │   └── pca_mean.npy
    ├── student/
    │   ├── acoustic_encoder.pt
    │   └── training_curve.png
    └── export/
        ├── acoustic_encoder.onnx
        └── acoustic_encoder.onnx.data
````` 

## Files

| File | Role |
|---|---|
| `extract_embeddings.py` | Stage 1. Runs YAMNet on FSD50K audio chunks and fits PCA on the resulting teacher embeddings. |
| `compute_mels.py` | Stage 2. Builds the aligned log-mel cache used as student input during distillation training. |
| `cnn.py` | Defines `AcousticEncoder`, the frozen CNN embedder trained by distillation. Input: `(1, 64, 61)` log-mel. Output: `32D` embedding. |
| `training_pipeline.py` | Defines the training configuration, data bundle, and trainer that execute the optimisation workflow. |
| `train.py` | High-level entrypoint. Runs the cache-building stages, then trains the `AcousticEncoder`. |

## Inputs

| Input | Meaning |
|---|---|
| `data/fsd50k/FSD50K.eval_audio/` | Source audio used for distillation. |
| `data/yamnet/yamnet.tflite` | Teacher model used to extract embeddings. |

## Outputs

| Output | Meaning |
|---|---|
| `distillation/outputs/fsd50k_cache/eval_embeddings.npy` | Teacher embedding cache of shape `(N, 1024)`. |
| `distillation/outputs/fsd50k_cache/eval_mels.npy` | Student input cache of shape `(N, 1, 64, 61)`. |
| `distillation/outputs/pca/pca_components.npy` | PCA projection matrix of shape `(32, 1024)`. |
| `distillation/outputs/pca/pca_mean.npy` | PCA mean vector of shape `(1024,)`. |
| `distillation/outputs/student/acoustic_encoder.pt` | Best training checkpoint for the frozen embedder. |
| `distillation/outputs/student/training_curve.png` | Train/validation loss plot for the run. |

## Training Objective

For each cached FSD50K frame:
- input: log-mel spectrogram `(1, 64, 61)`
- target: PCA-projected YAMNet embedding `(32,)`

The student model is trained with mean-squared error:

`MSE(AcousticEncoder(log_mel), PCA(YAMNet_embedding))`

MIMII data is not used in this stage.

## Usage

Run the full distillation pipeline with:

```bash
python -m distillation.train
```

This entrypoint:
1. runs `extract_embeddings.py`,
2. runs `compute_mels.py`,
3. trains the `AcousticEncoder`.

If you want to rebuild only the intermediate caches, run the stage scripts directly:

```bash
python distillation/extract_embeddings.py
python distillation/compute_mels.py
```
