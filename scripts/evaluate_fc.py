"""
Evaluate f_c encoder quality by simulating the deployment pipeline.

Usage:
    python scripts/evaluate_fc.py --checkpoint outputs/fc/conv/best.pt
    python scripts/evaluate_fc.py --checkpoint outputs/fc/tcn/best.pt

For each machine:
  1. Compute centroid from normal clip embeddings (simulates on-device calibration)
  2. Score all clips by distance to centroid (simulates on-device scoring)
  3. Compute AUC (measures anomaly detection performance)

Also computes separation ratio, silhouette score, and generates all plots.
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_dataloaders, load_config
from src.fc.encoders import build_encoder
from src.evaluation.metrics import evaluate_encoder, print_evaluation
from src.evaluation.visualisation import (
    plot_tsne, plot_distance_distributions, plot_pca_variance
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate f_c encoder")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", default="configs/fc.yaml", help="Config file")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()

    # Load config and checkpoint
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Restore encoder type from saved config
    saved_config = checkpoint.get("config", config)
    config["model"]["encoder"] = saved_config["model"]["encoder"]

    # Select device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Encoder: {config['model']['encoder']}")
    print(f"Device: {device}")
    print()

    # Build encoder and load weights
    encoder = build_encoder(config)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = encoder.to(device)

    # Build evaluation data
    print("Loading evaluation data...")
    _, eval_loader, _ = create_dataloaders(config)
    print()

    # Run evaluation
    results = evaluate_encoder(encoder, eval_loader, device)
    print_evaluation(results)

    # Generate plots
    output_dir = os.path.dirname(args.checkpoint)
    plot_tsne(results["clip_data"], output_dir)
    plot_distance_distributions(results["clip_data"], output_dir)
    plot_pca_variance(results["clip_data"], output_dir)


if __name__ == "__main__":
    main()
