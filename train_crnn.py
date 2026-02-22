#!/usr/bin/env python3
"""Training script for the standalone single-frame CRNN.

Uses only the HR images from the dataset — each HR image becomes an
independent (image, label) sample, producing 5× the number of tracks.

Architecture: CNN backbone → BiLSTM → CTC  (Shi et al., 2015)

Usage
-----
    # Default (32×128, batch 64, 30 epochs):
    python train_crnn.py

    # Custom:
    python train_crnn.py --epochs 50 --batch-size 128 --lr 1e-3

    # Submission mode (train on full data, predict test set):
    python train_crnn.py --submission-mode
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.data.single_frame_dataset import SingleFrameHRDataset
from src.models.CRNN import CRNN
from src.training.crnn_trainer import CRNNTrainer
from src.utils.common import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train standalone CRNN on HR images")
    p.add_argument("-n", "--experiment-name", type=str, default="crnn_standalone")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4, dest="learning_rate")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--img-height", type=int, default=32)
    p.add_argument("--img-width", type=int, default=128)
    p.add_argument("--rnn-hidden", type=int, default=256)
    p.add_argument("--map-to-seq-hidden", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--aug-level", type=str, choices=["full", "light"], default="full",
        help="Augmentation level for training data",
    )
    p.add_argument("--output-dir", type=str, default="results")
    p.add_argument(
        "--leaky-relu", action="store_true",
        help="Use LeakyReLU(0.2) instead of ReLU in CNN backbone",
    )
    p.add_argument(
        "--submission-mode", action="store_true",
        help="Train on full dataset and predict test set",
    )
    p.add_argument(
        "--decode-method", type=str, default="greedy",
        choices=["greedy", "beam_search", "prefix_beam_search"],
        help="CTC decoding method (only affects final test inference)",
    )
    p.add_argument("--beam-size", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- build config ----
    config = Config()
    config.EXPERIMENT_NAME = args.experiment_name
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.SEED = args.seed
    config.NUM_WORKERS = args.num_workers
    config.GRAD_CLIP = args.grad_clip
    config.IMG_HEIGHT = args.img_height
    config.IMG_WIDTH = args.img_width
    config.HIDDEN_SIZE = args.rnn_hidden
    config.WEIGHT_DECAY = args.weight_decay
    config.AUGMENTATION_LEVEL = args.aug_level
    config.OUTPUT_DIR = args.output_dir
    if args.data_root is not None:
        config.DATA_ROOT = args.data_root

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    seed_everything(config.SEED)

    # ---- validate input dims (CRNN assertions) ----
    assert config.IMG_HEIGHT % 16 == 0, (
        f"IMG_HEIGHT ({config.IMG_HEIGHT}) must be divisible by 16"
    )
    assert config.IMG_WIDTH % 4 == 0, (
        f"IMG_WIDTH ({config.IMG_WIDTH}) must be divisible by 4"
    )

    # ---- print config ----
    print("=" * 60)
    print("  CRNN Standalone Training (HR images only)")
    print("=" * 60)
    print(f"  EXPERIMENT     : {config.EXPERIMENT_NAME}")
    print(f"  DATA_ROOT      : {config.DATA_ROOT}")
    print(f"  IMG_SIZE       : {config.IMG_HEIGHT}×{config.IMG_WIDTH}")
    print(f"  EPOCHS         : {config.EPOCHS}")
    print(f"  BATCH_SIZE     : {config.BATCH_SIZE}")
    print(f"  LEARNING_RATE  : {config.LEARNING_RATE}")
    print(f"  RNN_HIDDEN     : {args.rnn_hidden}")
    print(f"  MAP_TO_SEQ     : {args.map_to_seq_hidden}")
    print(f"  LEAKY_RELU     : {args.leaky_relu}")
    print(f"  AUGMENTATION   : {config.AUGMENTATION_LEVEL}")
    print(f"  DEVICE         : {config.DEVICE}")
    print(f"  SUBMISSION_MODE: {args.submission_mode}")
    print("=" * 60)

    # ---- validate data path ----
    if not os.path.exists(config.DATA_ROOT):
        print(f"❌ ERROR: Data root not found: {config.DATA_ROOT}")
        sys.exit(1)

    # ---- common dataset kwargs ----
    ds_kwargs = dict(
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level=config.AUGMENTATION_LEVEL,
    )

    # ---- datasets ----
    if args.submission_mode:
        print("\n📌 SUBMISSION MODE — training on full dataset\n")
        train_ds = SingleFrameHRDataset(
            root_dir=config.DATA_ROOT, mode="train", full_train=True, **ds_kwargs
        )
        val_loader = None

        test_loader = None
        if os.path.exists(config.TEST_DATA_ROOT):
            test_ds = SingleFrameHRDataset(
                root_dir=config.TEST_DATA_ROOT,
                mode="val",
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
                collate_fn=SingleFrameHRDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
    else:
        train_ds = SingleFrameHRDataset(
            root_dir=config.DATA_ROOT, mode="train", **ds_kwargs
        )
        val_ds = SingleFrameHRDataset(
            root_dir=config.DATA_ROOT, mode="val", **ds_kwargs
        )
        val_loader = (
            DataLoader(
                val_ds,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                collate_fn=SingleFrameHRDataset.collate_fn,
                num_workers=config.NUM_WORKERS,
                pin_memory=True,
            )
            if len(val_ds) > 0
            else None
        )
        test_loader = None

    if len(train_ds) == 0:
        print("❌ Training dataset is empty!")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=SingleFrameHRDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # ---- model ----
    model = CRNN(
        img_channel=3,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        num_class=config.NUM_CLASSES,
        map_to_seq_hidden=args.map_to_seq_hidden,
        rnn_hidden=args.rnn_hidden,
        leaky_relu=args.leaky_relu,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 CRNN: {total_params:,} total params, {trainable:,} trainable")

    # ---- trainer ----
    trainer = CRNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        idx2char=config.IDX2CHAR,
    )

    trainer.fit()

    # ---- test inference (submission mode) ----
    if args.submission_mode and test_loader is not None:
        print("\n" + "=" * 60)
        print("📝 GENERATING SUBMISSION FILE")
        print("=" * 60)

        best_path = os.path.join(
            config.OUTPUT_DIR, f"{config.EXPERIMENT_NAME}_best.pth"
        )
        if os.path.exists(best_path):
            print(f"📦 Loading best checkpoint: {best_path}")
            model.load_state_dict(
                torch.load(best_path, map_location=config.DEVICE)
            )

        import torch.nn.functional as F
        from src.models.CRNN import ctc_decode

        model.eval()
        results = []
        with torch.no_grad():
            for images, _, _, _, track_ids in test_loader:
                images = images.to(config.DEVICE)
                logits = model(images)
                log_probs = F.log_softmax(logits, dim=2)

                decoded = ctc_decode(
                    log_probs,
                    label2char=config.IDX2CHAR,
                    blank=0,
                    method=args.decode_method,
                    beam_size=args.beam_size,
                )
                for i, chars in enumerate(decoded):
                    text = "".join(chars) if isinstance(chars[0], str) else "".join(
                        config.IDX2CHAR.get(c, "") for c in chars
                    )
                    results.append(f"{track_ids[i]},{text}")

        out_file = os.path.join(
            config.OUTPUT_DIR, f"submission_{config.EXPERIMENT_NAME}_final.txt"
        )
        with open(out_file, "w") as f:
            f.write("\n".join(results))
        print(f"✅ Saved {len(results)} predictions → {out_file}")


if __name__ == "__main__":
    main()
