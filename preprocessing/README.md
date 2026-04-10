
# preprocessing/

This folder contains two kinds of files: architecture files and pipeline files. The architecture files implement reusable the preprocessing components: audio loading, mel-spectrogram construction, and YAMNet loading. The pipeline files use those shared components to run complete preprocessing stages, cache their outputs, and prepare the inputs needed by later parts of the repository.


## Folder Structure

```text
preprocessing/
|- loader.py
|- mel_spectrogram.py
|- yamnet_loading.py
|- extract_embeddings.py
|- compute_mels.py
|- split_mimii.py
|- train_student_pipeline.py
|- outputs/
|  |- fsd50k_cache/
|  |- pca/
|  |- mimii_splits/
|- README.md
````` 

## File Roles

### Architecture Files

| File | Role |
|---|---|
| `loader.py` | Shared audio-loading and chunking helpers. |
| `mel_spectrogram.py` | Shared mel-spectrogram construction helper used by the preprocessing pipeline. |
| `yamnet_loading.py` | Shared helper for loading the local YAMNet TFLite model. |

### Pipeline Files

| File | Role |
|---|---|
| `extract_embeddings.py` | Runs YAMNet on FSD50K audio chunks, caches teacher embeddings, and fits PCA. |
| `compute_mels.py` | Converts the same FSD50K audio chunks into mel-spectrogram tensors and caches them. |
| `split_mimii.py` | Creates a fixed MIMII split manifest for later separator training and inference. |
| `train_student_pipeline.py` | Lightweight orchestration script that runs the student preprocessing stages in sequence. |

## Outputs

This folder produces the cached arrays and manifests needed by the rest of the repository.

| Output | Meaning | Used by |
|---|---|---|
| `preprocessing/outputs/fsd50k_cache/eval_embeddings.npy` | Teacher embeddings of shape `(N_chunks, 1024)` | `distillation/train.py` |
| `preprocessing/outputs/pca/pca_components.npy` | PCA projection matrix of shape `(32, 1024)` | `distillation/train.py` |
| `preprocessing/outputs/pca/pca_mean.npy` | PCA mean vector of shape `(1024,)` | `distillation/train.py` |
| `preprocessing/outputs/fsd50k_cache/eval_mels.npy` | Student inputs of shape `(N_chunks, 1, 64, 61)` | `distillation/train.py` |
| `preprocessing/outputs/mimii_splits/splits.json` | Fixed MIMII train/test manifest | `separator/train.py`, `inference/run.py` |

## How This Integrates With the Repo

The FSD50K preprocessing path supports knowledge distillation. `extract_embeddings.py` builds the teacher targets, `compute_mels.py` builds the aligned student inputs, and `distillation/train.py` then trains the student model from those cached outputs. The key alignment invariant is that mel tensor `i` corresponds to teacher embedding `i`.

The MIMII split path supports downstream anomaly detection. `split_mimii.py` does not move audio files; it only writes a reproducible JSON manifest describing which clips belong to training and evaluation. That manifest is then consumed by `separator/train.py` and `inference/run.py`.

## Usage

Run the distillation preprocessing stages with:

```bash
python -m preprocessing.train_student_pipeline
