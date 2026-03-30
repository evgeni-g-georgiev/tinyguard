#!/usr/bin/env python3
"""
train_student.py — Distil YAMNet into AcousticEncoder via MSE on FSD50K.

For each FSD50K frame:
  - Input:  log-mel spectrogram  (1, 64, 61)
  - Target: YAMNet embedding projected to 16D via PCA

Loss: MSE(student_output, pca_projected_teacher)
The student learns to reproduce the teacher's compressed representation
without ever seeing MIMII data.

Outputs
-------
  outputs/student/acoustic_encoder.pt   — best checkpoint (by val loss)
  outputs/student/training_curve.png    — train / val MSE loss curves

Prerequisites
-------------
    python scripts/prepare_teacher.py
    python scripts/prepare_mels.py

Usage
-----
    python scripts/train_student.py [--epochs 50] [--batch 256] [--lr 1e-3]
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.cnn import AcousticEncoder

# ── config ─────────────────────────────────────────────────────────────────

CACHE_DIR  = "outputs/fsd50k_cache"
PCA_DIR    = "outputs/pca"
OUTPUT_DIR = "outputs/student"


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=50)
    parser.add_argument("--batch",     type=int,   default=256)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--wd",        type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--val_frac",  type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load cached data ───────────────────────────────────────────────────
    for path, name in [
        (os.path.join(CACHE_DIR, "eval_mels.npy"),        "Mel spectrograms"),
        (os.path.join(CACHE_DIR, "eval_embeddings.npy"),  "Teacher embeddings"),
        (os.path.join(PCA_DIR,   "pca_components.npy"),   "PCA components"),
        (os.path.join(PCA_DIR,   "pca_mean.npy"),         "PCA mean"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {name} not found at {path}")
            print("Run prepare_teacher.py then prepare_mels.py first.")
            sys.exit(1)

    print("Loading data …")
    mels       = np.load(os.path.join(CACHE_DIR, "eval_mels.npy"))         # (N, 1, 64, 61)
    embeddings = np.load(os.path.join(CACHE_DIR, "eval_embeddings.npy"))   # (N, 1024)
    pca_comp   = np.load(os.path.join(PCA_DIR,   "pca_components.npy"))    # (16, 1024)
    pca_mean   = np.load(os.path.join(PCA_DIR,   "pca_mean.npy"))          # (1024,)
    print(f"  {mels.shape[0]:,} frames loaded")

    # Project teacher embeddings to 16D — these are the distillation targets
    targets = (embeddings - pca_mean) @ pca_comp.T   # (N, 16)

    # ── Dataset ────────────────────────────────────────────────────────────
    X = torch.tensor(mels,    dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    n_total = len(X)
    n_val   = max(1, int(n_total * args.val_frac))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(
        TensorDataset(X, y), [n_train, n_val], generator=generator
    )
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_set,   batch_size=args.batch,
                              num_workers=4, pin_memory=(device.type == "cuda"))

    print(f"  Train: {n_train:,}  Val: {n_val:,}  Batch: {args.batch}")

    # ── Model ──────────────────────────────────────────────────────────────
    model     = AcousticEncoder().to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    print(f"  AcousticEncoder: {n_params:,} parameters")

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  "
          f"{'LR':>8}  {'Best':>5}  {'Time':>6}")
    print(f"{'─' * 65}")

    best_val_loss = float("inf")
    best_epoch    = 0
    train_losses  = []
    val_losses    = []
    ckpt_path     = os.path.join(OUTPUT_DIR, "acoustic_encoder.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(Xb)
        train_loss = running / n_train
        train_losses.append(train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_running = sum(
                F.mse_loss(model(Xb.to(device)), yb.to(device)).item() * len(Xb)
                for Xb, yb in val_loader
            )
        val_loss = val_running / n_val
        val_losses.append(val_loss)

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_loss":         val_loss,
                "train_loss":       train_loss,
                "embedding_dim":    32,
                "param_count":      n_params,
                "args":             vars(args),
            }, ckpt_path)

        marker = " ✓" if is_best else ""
        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
              f"{lr_now:>8.2e}  {best_epoch:>5}  {elapsed:>5.1f}s{marker}")

    print(f"{'─' * 65}")
    print(f"\nBest epoch {best_epoch}  —  val MSE {best_val_loss:.5f}")
    print(f"Checkpoint → {ckpt_path}")

    # ── Training curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train", linewidth=1.5)
    ax.plot(val_losses,   label="Val",   linewidth=1.5)
    ax.axvline(best_epoch - 1, color="gray", linestyle="--",
               linewidth=1, label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("AcousticEncoder — Knowledge Distillation")
    ax.legend()
    ax.grid(alpha=0.3)
    curve_path = os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Training curve → {curve_path}")


if __name__ == "__main__":
    main()
