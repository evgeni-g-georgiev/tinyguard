"""
train.py — Run the full AcousticEncoder distillation pipeline.

First prepares the cached FSD50K inputs required for distillation by running
the teacher-embedding and log-mel preprocessing stages. Then trains the
AcousticEncoder to reproduce PCA-projected YAMNet embeddings using MSE loss.

For each cached FSD50K frame:
  - Input:  log-mel spectrogram  (1, 64, 61)
  - Target: YAMNet embedding projected to 32D via PCA

Outputs
-------
  distillation/outputs/fsd50k_cache/eval_embeddings.npy
  distillation/outputs/fsd50k_cache/eval_mels.npy
  distillation/outputs/pca/pca_components.npy
  distillation/outputs/pca/pca_mean.npy
  distillation/outputs/student/acoustic_encoder.pt
  distillation/outputs/student/training_curve.png

Usage
-----
  python distillation/train.py
  python distillation/train.py --epochs 50 --batch 256 --lr 1e-3
"""


import argparse

from distillation.extract_embeddings import extract_embeddings
from distillation.compute_mels import compute_mels
from distillation.training_pipeline import TrainConfig, DistillationTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    extract_embeddings()
    compute_mels()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    trainer = DistillationTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()


