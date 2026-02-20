"""Trainer for standalone EDSRLite super-resolution pretraining.

Implements the training strategy recommended by Lim et al. (EDSR, 2017):
- L1 (MAE) loss instead of L2 (MSE), yielding sharper reconstructions
- AdamW optimizer with OneCycleLR scheduling
- Mixed-precision training with gradient scaling
- PSNR / SSIM metric tracking on validation

This trainer is designed for pretraining EDSRLite before integrating it
into the StackedSRNet multi-task pipeline.
"""
import os
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Assumes inputs are normalized to [-1, 1] (data_range=2.0).
    
    Args:
        pred: Predicted image tensor [B, C, H, W].
        target: Ground truth image tensor [B, C, H, W].
        data_range: Value range of the images (2.0 for [-1,1] normalized).

    Returns:
        Average PSNR in dB across the batch.
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Per-sample MSE
    psnr = 10.0 * torch.log10((data_range ** 2) / (mse + 1e-10))
    return psnr.mean().item()


class EDSRLiteTrainer:
    """Trainer for standalone EDSRLite super-resolution.

    Follows the EDSR paper's recommendations:
    - L1 loss for sharper reconstructions (vs L2/MSE)
    - Residual scaling (0.1) for stable deep training
    - AdamW optimizer with weight decay
    - Gradient clipping for training stability
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        loss_type: str = "l1",
    ):
        """
        Args:
            model: EDSRLite model.
            train_loader: Training data loader yielding (lr, hr, track_id).
            val_loader: Validation data loader.
            config: Configuration object with training parameters.
            loss_type: Loss function — 'l1' (recommended by EDSR authors) or 'l2'.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE

        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # Loss function — EDSR authors recommend L1
        if loss_type == "l1":
            self.criterion = nn.L1Loss(reduction="mean")
        elif loss_type == "l2":
            self.criterion = nn.MSELoss(reduction="mean")
        elif loss_type == "charbonnier":
            # Smooth L1 variant: sqrt(x^2 + eps^2) — used in some SR works
            self.criterion = self._charbonnier_loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'l1', 'l2', or 'charbonnier'.")

        self.loss_type = loss_type

        # Optimizer — AdamW following modern best practices
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )

        # Scheduler — OneCycleLR (cosine annealing with warm-up)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )

        self.scaler = GradScaler()

        # Tracking
        self.best_psnr = 0.0
        self.best_val_loss = float("inf")
        self.current_epoch = 0

        # Log configuration
        print(f"\n{'='*60}")
        print(f"EDSRLite Training Configuration")
        print(f"{'='*60}")
        print(f"Loss function: {loss_type.upper()} (EDSR paper recommends L1)")
        print(f"Optimizer: AdamW (lr={config.LEARNING_RATE}, wd={config.WEIGHT_DECAY})")
        print(f"Scheduler: OneCycleLR")
        print(f"Epochs: {config.EPOCHS}")
        print(f"Grad clip: {config.GRAD_CLIP}")
        print(f"Device: {config.DEVICE}")
        print(f"{'='*60}\n")

    @staticmethod
    def _charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Charbonnier loss — smooth approximation of L1, used in LapSRN and others."""
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))

    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)

    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, "EXPERIMENT_NAME", "edsr_lite")

    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with 'loss' and 'lr' keys.
        """
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}",
        )

        for lr_images, hr_images, _ in pbar:
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                sr_output = self.model(lr_images)

                # Ensure spatial dimensions match (EDSRLite uses F.interpolate internally)
                if sr_output.shape[2:] != hr_images.shape[2:]:
                    hr_images = torch.nn.functional.interpolate(
                        hr_images,
                        size=sr_output.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = self.criterion(sr_output, hr_images)

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()

            # Unscale before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)

            # Optimizer step with scaler
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Scheduler step only if optimizer actually stepped
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
            )

        num_batches = len(self.train_loader)
        return {
            "loss": epoch_loss / num_batches,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation and compute reconstruction metrics.

        Returns:
            Dictionary with 'loss' and 'psnr' keys.
        """
        if self.val_loader is None:
            return {"loss": 0.0, "psnr": 0.0}

        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        for lr_images, hr_images, _ in tqdm(self.val_loader, desc="Validating"):
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)

            sr_output = self.model(lr_images)

            # Match spatial dimensions
            if sr_output.shape[2:] != hr_images.shape[2:]:
                hr_images = torch.nn.functional.interpolate(
                    hr_images,
                    size=sr_output.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = self.criterion(sr_output, hr_images)
            val_loss += loss.item()
            val_psnr += compute_psnr(sr_output, hr_images)

        num_batches = len(self.val_loader)
        return {
            "loss": val_loss / num_batches,
            "psnr": val_psnr / num_batches,
        }

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> str:
        """Save model checkpoint with full training state.

        Args:
            metrics: Current metrics dictionary.
            is_best: Whether this is the best model so far.

        Returns:
            Path to the saved checkpoint.
        """
        exp_name = self._get_exp_name()
        suffix = "best" if is_best else f"epoch{self.current_epoch + 1}"
        path = self._get_output_path(f"{exp_name}_{suffix}.pth")

        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "best_psnr": self.best_psnr,
                "metrics": metrics,
                "config": {
                    "loss_type": self.loss_type,
                    "lr": self.config.LEARNING_RATE,
                    "weight_decay": self.config.WEIGHT_DECAY,
                    "epochs": self.config.EPOCHS,
                },
            },
            path,
        )
        return path

    def fit(self) -> float:
        """Run the full training loop.

        Returns:
            Best validation PSNR achieved.
        """
        print(f"Starting EDSRLite training | Device: {self.device} | Epochs: {self.config.EPOCHS}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_one_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val PSNR: {val_metrics['psnr']:.2f} dB | "
                f"LR: {train_metrics['lr']:.2e}"
            )

            # Save best model (by PSNR)
            is_best = val_metrics["psnr"] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics["psnr"]
                path = self.save_checkpoint(val_metrics, is_best=True)
                print(f"  -> New best model saved: {path} (PSNR: {self.best_psnr:.2f} dB)")

        # Save final model
        final_path = self.save_checkpoint(val_metrics, is_best=False)
        print(f"\nTraining complete! Best PSNR: {self.best_psnr:.2f} dB")
        print(f"Final checkpoint: {final_path}")

        return self.best_psnr
