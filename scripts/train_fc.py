"""
Train the f_c encoder using self-supervised contrastive learning.

Usage:
    python scripts/train_fc.py --encoder conv --device auto
    python scripts/train_fc.py --encoder tcn --config configs/fc.yaml

The encoder learns general audio features from mel-spectrograms by pulling
augmented views of the same sound together and pushing different sounds apart.
No labels are used. The trained encoder is saved for downstream use by f_s.
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_dataloaders, load_config
from src.fc.contrastive import ContrastiveModel
from src.training.trainer import ContrastiveTrainer
from src.evaluation.visualisation import plot_training_curves


def select_device(requested: str) -> torch.device:
    """Select compute device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Train f_c encoder (contrastive)")
    parser.add_argument("--encoder", choices=["conv", "tcn"], default="conv",
                        help="Encoder architecture")
    parser.add_argument("--config", default="configs/fc.yaml",
                        help="Path to config file")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--output", default="outputs/fc",
                        help="Output directory for checkpoints and plots")
    args = parser.parse_args()

    config = load_config(args.config)
    config["model"]["encoder"] = args.encoder

    device = select_device(args.device)
    output_dir = os.path.join(args.output, args.encoder)

    print(f"Encoder: {args.encoder}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # Build data
    print("Loading data...")
    train_loader, eval_loader, stats = create_dataloaders(config)
    print()

    # Save normalisation stats (needed for deployment)
    stats_path = os.path.join(output_dir, "norm_stats.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"Normalisation stats saved to {stats_path}")

    # Build model
    print("\nBuilding model...")
    model = ContrastiveModel(config)
    print()

    # Train
    trainer = ContrastiveTrainer(model, config, device, output_dir)
    history = trainer.train(train_loader, eval_loader)

    # Plot training curves
    plot_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
