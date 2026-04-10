"""
training_pipeline.py — Run the AcousticEncoder distillation workflow.

Defines the configuration, data containers, and trainer used to fit the
AcousticEncoder on cached FSD50K distillation data. 

The model architecture lives in distillation/cnn.py.
"""


import sys
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import FSDCACHE_DIR, PCA_DIR, STUDENT_DIR
from distillation.cnn import AcousticEncoder


class TrainConfig:
    """Store the hyperparameters for distillation training."""
    def __init__(self, epochs=50, batch_size=256, lr=1e-3,
                 weight_decay=1e-4, val_frac=0.1, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.seed = seed


class DistillationDataBundle:
    """Store the loaded caches and projected targets.

    Args:
        mels: Log-mel spectrogram cache of shape (N, 1, 64, 61).
        teacher_embeddings: Teacher embedding cache of shape (N, 1024).
        pca_components: PCA projection matrix of shape (32, 1024).
        pca_mean: PCA mean vector of shape (1024,).
        targets: Projected teacher targets of shape (N, 32).
    """
    def __init__(self, mels, teacher_embeddings, pca_components, pca_mean, targets):
        self.mels = mels
        self.teacher_embeddings = teacher_embeddings
        self.pca_components = pca_components
        self.pca_mean = pca_mean
        self.targets = targets



class DistillationTrainer:
    """Run the AcousticEncoder distillation stage."""
    def __init__(self, config):
        self.config = config
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.checkpoint_path = STUDENT_DIR / "acoustic_encoder.pt"
        self.curve_path = STUDENT_DIR / "training_curve.png"

    def prepare_inputs(self):
        """Check that the required distillation caches already exist."""
        required = [
            (FSDCACHE_DIR / "eval_mels.npy", "Mel spectrograms"),
            (FSDCACHE_DIR / "eval_embeddings.npy", "Teacher embeddings"),
            (PCA_DIR / "pca_components.npy", "PCA components"),
            (PCA_DIR / "pca_mean.npy", "PCA mean"),
        ]

        for path, name in required:
            if not path.exists():
                print(f"ERROR: {name} not found at {path}")
                print(
                    "Run distillation/extract_embeddings.py then "
                    "distillation/compute_mels.py first."
                )
                sys.exit(1)

    def load_data_bundle(self):
        """Load the cached arrays and project the teacher targets.

        Returns:
            DistillationDataBundle: Bundle containing: (1) the mel cache, (2) teacher
            embeddings, (3) PCA data, and (4) projected targets.
        """
        print("Loading data ...")
        mels = np.load(FSDCACHE_DIR / "eval_mels.npy")
        teacher_embeddings = np.load(FSDCACHE_DIR / "eval_embeddings.npy")
        pca_components = np.load(PCA_DIR / "pca_components.npy")
        pca_mean = np.load(PCA_DIR / "pca_mean.npy")

        print(f"  {mels.shape[0]:,} frames loaded")

        targets = (teacher_embeddings - pca_mean) @ pca_components.T

        return DistillationDataBundle(
            mels=mels,
            teacher_embeddings=teacher_embeddings,
            pca_components=pca_components,
            pca_mean=pca_mean,
            targets=targets.astype(np.float32),
        )

    def build_dataloaders(self, bundle):
        """Split the dataset and build the training dataloaders.

        Args:
            bundle: Loaded distillation data.

        Returns:
            tuple: Training loader, validation loader, training size, and
            validation size.
        """
        X = torch.tensor(bundle.mels, dtype=torch.float32)
        y = torch.tensor(bundle.targets, dtype=torch.float32)

        n_total = len(X)
        n_val = max(1, int(n_total * self.config.val_frac))
        n_train = n_total - n_val

        generator = torch.Generator().manual_seed(self.config.seed)
        train_set, val_set = random_split(
            TensorDataset(X, y),
            [n_train, n_val],
            generator=generator,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size,
            num_workers=4,
            pin_memory=(self.device.type == "cuda"),
        )

        return train_loader, val_loader, n_train, n_val

    def build_model(self):
        """Create the model, optimiser, scheduler, and parameter count.

        Returns:
            tuple: AcousticEncoder, AdamW optimiser, cosine scheduler, and
            total parameter count.
        """
        model = AcousticEncoder().to(self.device)
        n_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )

        return model, optimizer, scheduler, n_params

    def train_one_epoch(self, model, train_loader, optimizer, n_train):
        """Run one optimisation epoch.

        Args:
            model: AcousticEncoder being trained.
            train_loader: DataLoader over the training split.
            optimizer: Optimiser used to update the model.
            n_train: Number of samples in the training split.

        Returns:
            float: Mean training loss for the epoch.
        """
        model.train()
        running = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)

            optimizer.zero_grad()
            loss = F.mse_loss(model(Xb), yb)
            loss.backward()
            optimizer.step()

            running += loss.item() * len(Xb)

        return running / n_train

    def validate_one_epoch(self, model, val_loader, n_val):
        """Evaluate the model on the validation split.

        Args:
            model: AcousticEncoder being evaluated.
            val_loader: DataLoader over the validation split.
            n_val: Number of samples in the validation split.

        Returns:
            float: Mean validation loss for the epoch.
        """
        model.eval()

        with torch.no_grad():
            val_running = sum(
                F.mse_loss(model(Xb.to(self.device)), yb.to(self.device)).item() * len(Xb)
                for Xb, yb in val_loader
            )

        return val_running / n_val

    def save_checkpoint(self, model, epoch, train_loss, val_loss, n_params):
        """Save the best checkpoint for the current run.

        Args:
            model: AcousticEncoder to save.
            epoch: Epoch at which the checkpoint was produced.
            train_loss: Mean training loss for the epoch.
            val_loss: Mean validation loss for the epoch.
            n_params: Total number of model parameters.
        """
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "embedding_dim": 32,
                "param_count": n_params,
                "args": {
                    "epochs": self.config.epochs,
                    "batch": self.config.batch_size,
                    "lr": self.config.lr,
                    "wd": self.config.weight_decay,
                    "val_frac": self.config.val_frac,
                    "seed": self.config.seed,
                },
            },
            self.checkpoint_path,
        )

    def save_training_curve(self, train_losses, val_losses, best_epoch):
        """Write the train/validation loss curve to disk.

        Args:
            train_losses: Training loss for each epoch.
            val_losses: Validation loss for each epoch.
            best_epoch: Epoch with the lowest validation loss.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(train_losses, label="Train", linewidth=1.5)
        ax.plot(val_losses, label="Val", linewidth=1.5)
        ax.axvline(
            best_epoch - 1,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Best (epoch {best_epoch})",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title("AcousticEncoder - Knowledge Distillation")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.curve_path, dpi=150)
        plt.close()

        print(f"Training curve -> {self.curve_path}")

    def train(self):
        """Run the full distillation training workflow."""
        torch.manual_seed(self.config.seed)
        print(f"Device: {self.device}")

        STUDENT_DIR.mkdir(parents=True, exist_ok=True)

        self.prepare_inputs()
        bundle = self.load_data_bundle()
        train_loader, val_loader, n_train, n_val = self.build_dataloaders(bundle)

        print(f"  Train: {n_train:,}  Val: {n_val:,}  Batch: {self.config.batch_size}")

        model, optimizer, scheduler, n_params = self.build_model()
        print(f"  AcousticEncoder: {n_params:,} parameters")

        print(f"\n{'-' * 65}")
        print(
            f"  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  "
            f"{'LR':>8}  {'Best':>5}  {'Time':>6}"
        )
        print(f"{'-' * 65}")

        best_val_loss = float("inf")
        best_epoch = 0
        train_losses = []
        val_losses = []

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            train_loss = self.train_one_epoch(model, train_loader, optimizer, n_train)
            val_loss = self.validate_one_epoch(model, val_loader, n_val)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step()
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                self.save_checkpoint(model, epoch, train_loss, val_loss, n_params)

            marker = " *" if is_best else ""
            print(
                f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
                f"{lr_now:>8.2e}  {best_epoch:>5}  {elapsed:>5.1f}s{marker}"
            )

        print(f"{'-' * 65}")
        print(f"\nBest epoch {best_epoch}  -  val MSE {best_val_loss:.5f}")
        print(f"Checkpoint -> {self.checkpoint_path}")

        self.save_training_curve(train_losses, val_losses, best_epoch)
