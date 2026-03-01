#!/usr/bin/env python3
"""Main entry point for OCR training pipeline."""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.dataset import MultiFrameDataset
from src.models.crnn import MultiFrameCRNN
from src.models.restran import ResTranOCR
from src.models.temptran import AlignedResNetTransOCR
from src.training.trainer import Trainer
from src.utils.common import seed_everything


def _build_model(config) -> torch.nn.Module:
    """Instantiate model based on config and move to device."""
    if config.MODEL_TYPE == "temptran":
        model = AlignedResNetTransOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        )
    elif config.MODEL_TYPE == "restran":
        model = ResTranOCR(
            num_classes=config.NUM_CLASSES,
            transformer_heads=config.TRANSFORMER_HEADS,
            transformer_layers=config.TRANSFORMER_LAYERS,
            transformer_ff_dim=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            use_stn=config.USE_STN,
        )
    else:
        model = MultiFrameCRNN(
            num_classes=config.NUM_CLASSES,
            hidden_size=config.HIDDEN_SIZE,
            rnn_dropout=config.RNN_DROPOUT,
            use_stn=config.USE_STN,
        )
    return model.to(config.DEVICE)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Frame OCR for License Plate Recognition"
    )
    parser.add_argument(
        "-n", "--experiment-name", type=str, default=None,
        help="Experiment name for checkpoint/submission files (default: from config)"
    )
    parser.add_argument(
        "-m", "--model", type=str, choices=["crnn", "restran", "temptran"], default=None,
        help="Model architecture: 'crnn', 'restran', or 'temptran' (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="learning_rate",
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Root directory for training data (default: from config)"
    )
    parser.add_argument(
        "--test-data-root", type=str, default=None,
        help="Root directory for test data (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: from config)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of data loader workers (default: from config)"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=None,
        help="Gradient clipping value (default: from config)"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=None,
        help="LSTM hidden size for CRNN (default: from config)"
    )
    parser.add_argument(
        "--transformer-heads", type=int, default=None,
        help="Number of transformer attention heads (default: from config)"
    )
    parser.add_argument(
        "--transformer-layers", type=int, default=None,
        help="Number of transformer encoder layers (default: from config)"
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=None,
        help="Dropout rate for transformer layers (default: from config)",
    )
    parser.add_argument(
        "--aug-level",
        type=str,
        choices=["full", "light"],
        default=None,
        help="Augmentation level for training data (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save checkpoints and submission files (default: results/)",
    )
    parser.add_argument(
        "--no-stn",
        action="store_true",
        help="Disable Spatial Transformer Network (STN) alignment",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint (.pth) to load before training",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip training; load --pretrain checkpoint and generate submission only",
    )
    parser.add_argument(
        "--submission-mode",
        action="store_true",
        help="Train on full dataset and generate submission file for test data",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()
    
    # Initialize config with CLI overrides
    config = Config()
    
    # Map CLI arguments to config attributes
    arg_to_config = {
        'experiment_name': 'EXPERIMENT_NAME',
        'model': 'MODEL_TYPE',
        'epochs': 'EPOCHS',
        'batch_size': 'BATCH_SIZE',
        'learning_rate': 'LEARNING_RATE',
        'data_root': 'DATA_ROOT',
        'test_data_root': 'TEST_DATA_ROOT',
        'seed': 'SEED',
        'num_workers': 'NUM_WORKERS',
        'grad_clip': 'GRAD_CLIP',
        'hidden_size': 'HIDDEN_SIZE',
        'transformer_heads': 'TRANSFORMER_HEADS',
        'transformer_layers': 'TRANSFORMER_LAYERS',
        'transformer_dropout': 'TRANSFORMER_DROPOUT',
    }
    
    for arg_name, config_name in arg_to_config.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(config, config_name, value)
    
    # Special cases
    if args.aug_level is not None:
        config.AUGMENTATION_LEVEL = args.aug_level
    
    if args.no_stn:
        config.USE_STN = False
    
    # Output directory
    config.OUTPUT_DIR = args.output_dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    seed_everything(config.SEED)
    
    print(f"🚀 Configuration:")
    print(f"   EXPERIMENT: {config.EXPERIMENT_NAME}")
    print(f"   MODEL: {config.MODEL_TYPE}")
    print(f"   USE_STN: {config.USE_STN}")
    print(f"   DATA_ROOT: {config.DATA_ROOT}")
    print(f"   EPOCHS: {config.EPOCHS}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   DEVICE: {config.DEVICE}")
    print(f"   SUBMISSION_MODE: {args.submission_mode}")
    if args.predict_only:
        print(f"   PREDICT_ONLY: True")
        print(f"   TEST_DATA_ROOT: {config.TEST_DATA_ROOT}")
        print(f"   PRETRAIN: {args.pretrain}")
    
    # Validate data path (skip for predict-only mode)
    if args.predict_only:
        if not args.pretrain:
            print("❌ ERROR: --predict-only requires --pretrain <checkpoint>")
            sys.exit(1)
        if not os.path.exists(config.TEST_DATA_ROOT):
            print(f"❌ ERROR: Test data not found at {config.TEST_DATA_ROOT}")
            sys.exit(1)
    elif not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # ── Predict-only: skip all training data setup ──────────────
    if args.predict_only:
        # Build model
        model = _build_model(config)

        # Load checkpoint
        print(f"📦 Loading pretrained checkpoint: {args.pretrain}")
        state_dict = torch.load(args.pretrain, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
        print("✅ Pretrained weights loaded successfully")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model ({config.MODEL_TYPE}): {total_params:,} params")

        # Build test loader
        test_ds = MultiFrameDataset(
            root_dir=config.TEST_DATA_ROOT,
            mode='val',
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            char2idx=config.CHAR2IDX,
            seed=config.SEED,
            is_test=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=MultiFrameDataset.collate_fn,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        print("\n" + "="*60)
        print("📝 PREDICT-ONLY MODE — GENERATING SUBMISSION FILE")
        print("="*60)
        exp_name = config.EXPERIMENT_NAME
        trainer = Trainer(
            model=model,
            train_loader=test_loader,  # placeholder, unused
            val_loader=None,
            config=config,
            idx2char=config.IDX2CHAR,
        )
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")
        return

    # Common dataset parameters
    common_ds_params = {
        'split_ratio': config.SPLIT_RATIO,
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'char2idx': config.CHAR2IDX,
        'val_split_file': config.VAL_SPLIT_FILE,
        'seed': config.SEED,
        'augmentation_level': config.AUGMENTATION_LEVEL,
    }
    
    # Create datasets based on mode
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE ENABLED")
        print("   - Training on FULL dataset (no validation split)")
        print("   - Will generate predictions for test data after training\n")
        
        # Create training dataset with full_train=True
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            full_train=True,
            **common_ds_params
        )
        
        # Create test dataset if test data exists
        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = MultiFrameDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode='val',
                img_height=config.IMG_HEIGHT,
                img_width=config.IMG_WIDTH,
                char2idx=config.CHAR2IDX,
                seed=config.SEED,
                is_test=True,
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print(f"⚠️ WARNING: Test data not found at {config.TEST_DATA_ROOT}")
        
        val_loader = None
    else:
        # Normal training/validation split mode
        train_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='train',
            **common_ds_params
        )
        
        val_ds = MultiFrameDataset(
            root_dir=config.DATA_ROOT,
            mode='val',
            **common_ds_params
        )
        
        val_loader = None
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=MultiFrameDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True
            )
        else:
            print("⚠️ WARNING: Validation dataset is empty.")
        
        test_loader = None
    
    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    # Create training data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model based on config
    model = _build_model(config)
    
    # Load pretrained checkpoint if provided
    if args.pretrain:
        if os.path.isfile(args.pretrain):
            print(f"📦 Loading pretrained checkpoint: {args.pretrain}")
            state_dict = torch.load(args.pretrain, map_location=config.DEVICE)
            model.load_state_dict(state_dict)
            print("✅ Pretrained weights loaded successfully")
        else:
            print(f"❌ ERROR: Pretrained checkpoint not found: {args.pretrain}")
            sys.exit(1)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model ({config.MODEL_TYPE}): {total_params:,} total params, {trainable_params:,} trainable")

    # Initialize trainer and start training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR
    )
    
    trainer.fit()
    
    # Run test inference in submission mode
    if args.submission_mode and test_loader is not None:
        print("\n" + "="*60)
        print("📝 GENERATING SUBMISSION FILE")
        print("="*60)
        
        # Load best checkpoint if it exists
        exp_name = config.EXPERIMENT_NAME
        best_model_path = os.path.join(config.OUTPUT_DIR, f"{exp_name}_best.pth")
        if os.path.exists(best_model_path):
            print(f"📦 Loading best checkpoint: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        else:
            print("⚠️ No best checkpoint found, using final model weights")
        
        # Run inference on test data
        trainer.predict_test(test_loader, output_filename=f"submission_{exp_name}_final.txt")


if __name__ == "__main__":
    main()
