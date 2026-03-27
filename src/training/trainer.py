"""
Training loop for contrastive f_c learning.

Handles optimiser, LR scheduling, early stopping, checkpointing,
and training/validation loss tracking.
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class ContrastiveTrainer:
    """Trains a ContrastiveModel with NT-Xent loss."""

    def __init__(self, model, config: dict, device: torch.device, output_dir: str):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]

        self.optimiser = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=train_cfg["lr_scheduler"]["patience"],
            factor=train_cfg["lr_scheduler"]["factor"],
        )
        self.es_patience = train_cfg["early_stopping"]["patience"]

        self.history = {"train_loss": [], "val_loss": [], "lr": []}

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run full training loop.

        Args:
            train_loader: DataLoader yielding (batch,) tensors
            val_loader: DataLoader yielding (batch, metadata) tuples

        Returns:
            Training history dict
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate(val_loader)

            lr = self.optimiser.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)

            self.scheduler.step(val_loss)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr:.6f} | {elapsed:.1f}s"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                patience_counter += 1

            if patience_counter >= self.es_patience:
                print(f"Early stopping at epoch {epoch} (patience={self.es_patience})")
                break

        self._save_checkpoint("final.pt")
        print(f"Best validation loss: {best_val_loss:.4f}")
        return self.history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc="Train", leave=False):
            batch = batch.to(self.device)
            output = self.model(batch)
            loss = output["loss"]

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        """Compute validation contrastive loss.

        The eval DataLoader yields (tensor, metadata) tuples.
        We only use the tensor for loss computation.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_data in loader:
            # EvaluationDataset returns (tensor, metadata)
            if isinstance(batch_data, (list, tuple)):
                batch = batch_data[0]
            else:
                batch = batch_data

            batch = batch.to(self.device)
            output = self.model(batch)

            total_loss += output["loss"].item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, filename: str):
        """Save encoder weights (without projection head) and full model."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "encoder_state_dict": self.model.encoder.state_dict(),
            "full_model_state_dict": self.model.state_dict(),
            "config": self.config,
            "history": self.history,
        }, path)
