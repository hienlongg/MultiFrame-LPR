#!/usr/bin/env python3
"""Training script for standalone EDSRLite super-resolution pretraining.

Trains the EDSRLite model on paired (LR, HR) license plate images using
L1 loss as recommended by the EDSR paper (Lim et al., CVPRW 2017).

This script is intended for pretraining the SR backbone before integrating
it into the StackedSRNet multi-task pipeline. The pretrained weights can be
loaded into StackedSRNet.sr_backbone for downstream OCR training.

Usage:
    # Default L1 training (EDSR-recommended)
    python train_edsr.py --epochs 50 --batch-size 32

    # With custom resolution targets
    python train_edsr.py --hr-height 48 --hr-width 124 --lr-height 16 --lr-width 64

    # Charbonnier loss (smooth L1 variant)
    python train_edsr.py --loss charbonnier --epochs 100

    # Load pretrained weights into StackedSRNet after training:
    #   checkpoint = torch.load("results/edsr_pretrain_best.pth")
    #   stacked_sr_net.sr_backbone.load_state_dict(checkpoint["model_state_dict"])
"""
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.sr_dataset import SRDataset
from src.models.EDSR.EDSRLite import EDSRLite
from src.training.edsr_trainer import EDSRLiteTrainer
from src.utils.common import seed_everything


@dataclass
class EDSRConfig(Config):
    """Configuration for standalone EDSRLite SR training."""

    # Experiment tracking
    MODEL_TYPE: str = "edsr_lite"
    EXPERIMENT_NAME: str = "edsr_pretrain"

    # EDSRLite architecture
    EDSR_NUM_FEATURES: int = 64
    EDSR_NUM_BLOCKS: int = 16
    EDSR_RES_SCALE: float = 0.1

    # Resolution configuration
    LR_HEIGHT: int = 16       # Input LR resolution
    LR_WIDTH: int = 64
    HR_HEIGHT: int = 48       # Target HR resolution (matches StackedSRNet target)
    HR_WIDTH: int = 124

    # Loss function — 'l1' (EDSR default), 'l2', or 'charbonnier'
    LOSS_TYPE: str = "l1"

    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4  # EDSR paper uses 1e-4
    EPOCHS: int = 50
    WEIGHT_DECAY: float = 0.0   # EDSR paper uses 0 weight decay
    GRAD_CLIP: float = 5.0

    # Data
    INCLUDE_SYNTHETIC: bool = True  # Include degraded HR as synthetic LR

    def __post_init__(self):
        """Compute derived attributes."""
        super().__post_init__()
        self.IMG_HEIGHT = self.LR_HEIGHT
        self.IMG_WIDTH = self.LR_WIDTH


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EDSRLite for Super-Resolution (EDSR pretraining)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoints",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate", help="Learning rate",
    )
    parser.add_argument(
        "--loss", type=str, choices=["l1", "l2", "charbonnier"], default=None,
        help="Loss function (EDSR paper recommends L1)",
    )

    # Architecture
    parser.add_argument("--num-features", type=int, default=None, help="EDSR feature channels")
    parser.add_argument("--num-blocks", type=int, default=None, help="EDSR residual blocks")
    parser.add_argument("--res-scale", type=float, default=None, help="Residual scaling factor")

    # Resolution
    parser.add_argument("--lr-height", type=int, default=None, help="LR input height")
    parser.add_argument("--lr-width", type=int, default=None, help="LR input width")
    parser.add_argument("--hr-height", type=int, default=None, help="HR target height")
    parser.add_argument("--hr-width", type=int, default=None, help="HR target width")

    # Data
    parser.add_argument("--data-root", type=str, default=None, help="Training data root")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--no-synthetic", action="store_true",
        help="Disable synthetic LR from degraded HR images",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser.parse_args()


def main():
    """Main entry point for EDSRLite training."""
    args = parse_args()
    config = EDSRConfig()

    # Apply CLI overrides
    overrides = {
        "experiment_name": "EXPERIMENT_NAME",
        "epochs": "EPOCHS",
        "batch_size": "BATCH_SIZE",
        "learning_rate": "LEARNING_RATE",
        "num_features": "EDSR_NUM_FEATURES",
        "num_blocks": "EDSR_NUM_BLOCKS",
        "res_scale": "EDSR_RES_SCALE",
        "lr_height": "LR_HEIGHT",
        "lr_width": "LR_WIDTH",
        "hr_height": "HR_HEIGHT",
        "hr_width": "HR_WIDTH",
        "data_root": "DATA_ROOT",
        "seed": "SEED",
        "num_workers": "NUM_WORKERS",
        "output_dir": "OUTPUT_DIR",
    }
    for arg_name, config_name in overrides.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)

    if args.loss is not None:
        config.LOSS_TYPE = args.loss
    if args.no_synthetic:
        config.INCLUDE_SYNTHETIC = False

    # Recompute derived attrs
    config.__post_init__()

    # Ensure output dir exists
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Seed
    seed_everything(config.SEED)

    # Print configuration
    print(f"\n{'='*60}")
    print(f"EDSRLite Super-Resolution Training")
    print(f"{'='*60}")
    print(f"Experiment:     {config.EXPERIMENT_NAME}")
    print(f"Loss:           {config.LOSS_TYPE.upper()}")
    print(f"LR resolution:  {config.LR_HEIGHT} x {config.LR_WIDTH}")
    print(f"HR resolution:  {config.HR_HEIGHT} x {config.HR_WIDTH}")
    print(f"Architecture:   {config.EDSR_NUM_FEATURES}ch, {config.EDSR_NUM_BLOCKS} blocks, "
          f"res_scale={config.EDSR_RES_SCALE}")
    print(f"Batch size:     {config.BATCH_SIZE}")
    print(f"Learning rate:  {config.LEARNING_RATE}")
    print(f"Epochs:         {config.EPOCHS}")
    print(f"Synthetic LR:   {config.INCLUDE_SYNTHETIC}")
    print(f"Device:         {config.DEVICE}")
    print(f"{'='*60}\n")

    # Validate data path
    if not os.path.exists(config.DATA_ROOT):
        print(f"ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # --- Data ---
    train_dataset = SRDataset(
        root_dir=config.DATA_ROOT,
        mode="train",
        split_ratio=config.SPLIT_RATIO,
        lr_height=config.LR_HEIGHT,
        lr_width=config.LR_WIDTH,
        hr_height=config.HR_HEIGHT,
        hr_width=config.HR_WIDTH,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        include_synthetic=config.INCLUDE_SYNTHETIC,
    )

    val_dataset = SRDataset(
        root_dir=config.DATA_ROOT,
        mode="val",
        split_ratio=config.SPLIT_RATIO,
        lr_height=config.LR_HEIGHT,
        lr_width=config.LR_WIDTH,
        hr_height=config.HR_HEIGHT,
        hr_width=config.HR_WIDTH,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        include_synthetic=False,  # No synthetic augmentation for val
    )

    if len(train_dataset) == 0:
        print("ERROR: Training dataset is empty.")
        sys.exit(1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=SRDataset.collate_fn,
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            collate_fn=SRDataset.collate_fn,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches:   {len(val_loader)}")

    # --- Model ---
    model = EDSRLite(
        in_channels=3,
        out_channels=3,
        num_features=config.EDSR_NUM_FEATURES,
        num_blocks=config.EDSR_NUM_BLOCKS,
        res_scale=config.EDSR_RES_SCALE,
        target_size=(config.HR_HEIGHT, config.HR_WIDTH),
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params, {trainable_params:,} trainable")

    # --- Resume from checkpoint ---
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded weights from epoch {checkpoint.get('epoch', '?')}")

    # --- Training ---
    trainer = EDSRLiteTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        loss_type=config.LOSS_TYPE,
    )

    try:
        best_psnr = trainer.fit()
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best PSNR: {best_psnr:.2f} dB")
        print(f"Checkpoints: {config.OUTPUT_DIR}/")
        print(f"{'='*60}")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Progress saved.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
