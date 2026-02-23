#!/usr/bin/env python3
"""Training script for LP-Diff diffusion-based super-resolution.

Usage:
    python train_lpdiff.py                           # train with defaults
    python train_lpdiff.py --phase val               # run validation / generate SR
    python train_lpdiff.py --lr 1e-4 --batch-size 8  # override hyper-params
    python train_lpdiff.py --resume checkpoint/I50000_E15  # resume training
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import LPDiffConfig, get_lpdiff_config
from src.data.lpdiff_dataset import create_dataset, create_dataloader
from src.models.LPDiff.model import DDPM
from src.utils.common import seed_everything
from src.utils.metrics import tensor2img, save_img, calculate_psnr, calculate_ssim

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str) -> logging.Logger:
    """Configure the ``base`` and ``val`` loggers."""
    os.makedirs(log_dir, exist_ok=True)

    # Base logger (train)
    base_logger = logging.getLogger('base')
    base_logger.setLevel(logging.INFO)
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    base_logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s'))
    base_logger.addHandler(ch)

    # Validation logger
    val_logger = logging.getLogger('val')
    val_logger.setLevel(logging.INFO)
    vfh = logging.FileHandler(os.path.join(log_dir, 'val.log'), mode='a')
    vfh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    val_logger.addHandler(vfh)

    return base_logger


# ---------------------------------------------------------------------------
# TensorBoard (optional)
# ---------------------------------------------------------------------------

def get_tb_writer(log_dir: str):
    """Return a TensorBoard SummaryWriter if tensorboardX is available."""
    try:
        from tensorboardX import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except ImportError:
            return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LP-Diff SR Training')
    p.add_argument('--phase', choices=['train', 'val'], default='train',
                   help='Run training or validation/generation')
    p.add_argument('--data-root', type=str, default=None,
                   help='Root directory of training data (default: config)')
    p.add_argument('--val-split', type=str, default='data/val_tracks.json',
                   help='Path to val split JSON file')
    p.add_argument('--resume', type=str, default=None,
                   help='Resume state prefix, e.g. checkpoint/I50000_E15')
    p.add_argument('--lr', type=float, default=None, help='Learning rate')
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--epochs', type=int, default=None, help='Total training epochs')
    p.add_argument('--val-freq', type=int, default=None,
                   help='Validate every N epochs (default: 5)')
    p.add_argument('--save-freq', type=int, default=None,
                   help='Save checkpoint every N epochs (default: 10)')
    p.add_argument('--print-freq', type=int, default=None,
                   help='Log every N iterations within an epoch')
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--height', type=int, default=None, help='Target image height')
    p.add_argument('--width', type=int, default=None, help='Target image width')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=str, default=None,
                   help='Comma-separated GPU ids, e.g. "0,1"')
    p.add_argument('--timesteps', type=int, default=None,
                   help='Number of diffusion timesteps')
    p.add_argument('--output-dir', type=str, default='experiments/lpdiff',
                   help='Base output directory')
    return p.parse_args()


def build_config(args: argparse.Namespace) -> LPDiffConfig:
    """Build LPDiffConfig from CLI arguments."""
    cfg = get_lpdiff_config(phase=args.phase)

    # GPU
    if args.gpu is not None:
        cfg.gpu_ids = [int(x) for x in args.gpu.split(',') if x.strip()]
    elif not torch.cuda.is_available():
        cfg.gpu_ids = []

    # Data
    if args.data_root:
        cfg.train_dataset.dataroot = args.data_root
        cfg.val_dataset.dataroot = args.data_root
    if args.batch_size is not None:
        cfg.train_dataset.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train_dataset.num_workers = args.num_workers
        cfg.val_dataset.num_workers = args.num_workers
    if args.height is not None:
        cfg.train_dataset.height = args.height
        cfg.val_dataset.height = args.height
    if args.width is not None:
        cfg.train_dataset.width = args.width
        cfg.val_dataset.width = args.width

    # Training
    if args.lr is not None:
        cfg.train.optimizer.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.val_freq is not None:
        cfg.train.val_freq = args.val_freq
    if args.save_freq is not None:
        cfg.train.save_checkpoint_freq = args.save_freq
    if args.print_freq is not None:
        cfg.train.print_freq = args.print_freq

    # Diffusion timesteps
    if args.timesteps is not None:
        cfg.beta_schedule_train.n_timestep = args.timesteps
        cfg.beta_schedule_val.n_timestep = args.timesteps

    # Resume
    if args.resume:
        cfg.path.resume_state = args.resume
        cfg.train.resume_training = True

    # Paths (relative to output dir)
    cfg.path.log = os.path.join(args.output_dir, 'logs')
    cfg.path.tb_logger = os.path.join(args.output_dir, 'tb_logger')
    cfg.path.results = os.path.join(args.output_dir, 'results')
    cfg.path.checkpoint = os.path.join(args.output_dir, 'checkpoint')

    return cfg


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: LPDiffConfig, args: argparse.Namespace) -> None:
    """Epoch-based training loop."""
    logger = setup_logging(cfg.path.log)
    logger.info('=' * 60)
    logger.info('LP-Diff Super-Resolution Training')
    logger.info('=' * 60)
    logger.info(f'Phase: {cfg.phase}')
    logger.info(f'GPU IDs: {cfg.gpu_ids}')
    logger.info(f'Image size: {cfg.train_dataset.height}x{cfg.train_dataset.width}')
    logger.info(f'Batch size: {cfg.train_dataset.batch_size}')
    logger.info(f'Learning rate: {cfg.train.optimizer.lr}')
    logger.info(f'Epochs: {cfg.train.epochs}')
    logger.info(f'Timesteps: {cfg.beta_schedule_train.n_timestep}')

    # Create output directories
    for d in [cfg.path.log, cfg.path.results, cfg.path.checkpoint, cfg.path.tb_logger]:
        os.makedirs(d, exist_ok=True)

    # TensorBoard
    tb_writer = get_tb_writer(cfg.path.tb_logger)

    # Datasets
    train_set = create_dataset(
        cfg.train_dataset, phase='train',
        val_split_file=args.val_split, seed=args.seed,
    )
    train_loader = create_dataloader(train_set, cfg.train_dataset, phase='train')

    val_set = create_dataset(
        cfg.val_dataset, phase='val',
        val_split_file=args.val_split, seed=args.seed,
    )
    val_loader = create_dataloader(val_set, cfg.val_dataset, phase='val')

    logger.info(f'Train samples: {len(train_set)}, Val samples: {len(val_set)}')

    # Model
    diffusion = DDPM(cfg)
    logger.info('Model initialised.')
    diffusion.print_network()

    # State
    global_step = diffusion.begin_step
    start_epoch = diffusion.begin_epoch
    total_epochs = cfg.train.epochs
    best_loss = float('inf')
    best_psnr = float('-inf')

    if cfg.path.resume_state:
        logger.info(f'Resumed from epoch {start_epoch}, step {global_step}.')

    # Set training noise schedule
    diffusion.set_new_noise_schedule(cfg.beta_schedule_train, schedule_phase='train')

    # ---- Epoch loop ----
    logger.info('Begin training...')
    for epoch in range(start_epoch + 1, total_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for train_data in train_loader:
            global_step += 1
            num_batches += 1

            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()

            logs = diffusion.get_current_log()
            epoch_loss += logs.get('l_pix', 0.0)

            # --- Per-iteration logging ---
            if global_step % cfg.train.print_freq == 0:
                msg = f'<epoch:{epoch:3d}, iter:{global_step:8,d}> '
                for k, v in logs.items():
                    msg += f'{k}: {v:.4e} '
                    if tb_writer:
                        tb_writer.add_scalar(k, v, global_step)
                logger.info(msg)

        # --- End-of-epoch summary ---
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            f'Epoch {epoch}/{total_epochs} complete — '
            f'avg l_pix: {avg_epoch_loss:.4e}, steps: {num_batches}'
        )
        if tb_writer:
            tb_writer.add_scalar('epoch/l_pix', avg_epoch_loss, epoch)

        # --- Validation (every val_freq epochs) ---
        if epoch % cfg.train.val_freq == 0:
            avg_psnr, avg_loss = _validate(
                diffusion, val_loader, cfg, epoch, global_step,
                logger, tb_writer,
            )

            # Best-model checkpointing
            if avg_loss <= best_loss and avg_psnr >= best_psnr:
                best_loss = avg_loss
                best_psnr = avg_psnr
                diffusion.save_best_both(epoch, global_step)
            elif avg_loss < best_loss:
                best_loss = avg_loss
                diffusion.save_best_loss(epoch, global_step)
            elif avg_psnr > best_psnr:
                best_psnr = avg_psnr
                diffusion.save_best_psnr(epoch, global_step)

            # Restore training noise schedule
            diffusion.set_new_noise_schedule(
                cfg.beta_schedule_train, schedule_phase='train')

        # --- Periodic checkpoint (every save_checkpoint_freq epochs) ---
        if epoch % cfg.train.save_checkpoint_freq == 0:
            logger.info('Saving checkpoint...')
            diffusion.save_network(epoch, global_step)

    logger.info('Training complete.')
    if tb_writer:
        tb_writer.close()


def _validate(
    diffusion: DDPM,
    val_loader,
    cfg: LPDiffConfig,
    epoch: int,
    step: int,
    logger: logging.Logger,
    tb_writer,
) -> tuple[float, float]:
    """Run a full validation pass: compute PSNR and save sample images."""
    result_path = os.path.join(cfg.path.results, str(epoch))
    os.makedirs(result_path, exist_ok=True)

    diffusion.set_new_noise_schedule(cfg.beta_schedule_val, schedule_phase='val')

    avg_psnr = 0.0
    avg_loss = 0.0
    count = 0

    for val_data in val_loader:
        count += 1
        diffusion.feed_data(val_data)
        loss = diffusion.test(continous=False)
        visuals = diffusion.get_current_visuals()

        sr_img = tensor2img(visuals['SR'])
        hr_img = tensor2img(visuals['HR'])

        # Save images
        save_img(sr_img, os.path.join(result_path, f'{step}_{count}_sr.png'))
        save_img(hr_img, os.path.join(result_path, f'{step}_{count}_hr.png'))
        for i in range(1, 6):
            key = f'LR{i}'
            if key in visuals:
                lr_img = tensor2img(visuals[key])
                save_img(lr_img, os.path.join(
                    result_path, f'{step}_{count}_lr{i}.png'))

        # TensorBoard grid
        if tb_writer:
            lr1_img = tensor2img(visuals['LR1'])
            lr3_img = tensor2img(visuals.get('LR3', visuals['LR1']))
            grid = np.concatenate([lr1_img, lr3_img, sr_img, hr_img], axis=1)
            tb_writer.add_image(
                f'Epoch_{epoch}',
                np.transpose(grid, [2, 0, 1]),
                count,
            )

        psnr = calculate_psnr(sr_img, hr_img)
        avg_psnr += psnr
        avg_loss += loss.item() if hasattr(loss, 'item') else float(loss)

    if count > 0:
        avg_psnr /= count
        avg_loss /= count

    logger.info(f'# Validation # PSNR: {avg_psnr:.4f}  Loss: {avg_loss:.4e}')
    val_logger = logging.getLogger('val')
    val_logger.info(
        f'<epoch:{epoch:3d}, step:{step:8,d}> psnr: {avg_psnr:.4e} loss: {avg_loss:.4e}'
    )

    if tb_writer:
        tb_writer.add_scalar('val/psnr', avg_psnr, epoch)
        tb_writer.add_scalar('val/loss', avg_loss, epoch)

    return avg_psnr, avg_loss


# ---------------------------------------------------------------------------
# Evaluation / generation
# ---------------------------------------------------------------------------

def evaluate(cfg: LPDiffConfig, args: argparse.Namespace) -> None:
    """Generate SR images from the validation set (``--phase val``)."""
    logger = setup_logging(cfg.path.log)
    logger.info('=' * 60)
    logger.info('LP-Diff Super-Resolution — Evaluation')
    logger.info('=' * 60)

    val_set = create_dataset(
        cfg.val_dataset, phase='val',
        val_split_file=args.val_split, seed=args.seed,
    )
    val_loader = create_dataloader(val_set, cfg.val_dataset, phase='val')

    # Model (loads weights from resume_state)
    diffusion = DDPM(cfg)
    logger.info('Model loaded.')

    diffusion.set_new_noise_schedule(cfg.beta_schedule_val, schedule_phase='val')

    result_path = cfg.path.results
    os.makedirs(result_path, exist_ok=True)

    avg_psnr = 0.0
    avg_ssim = 0.0
    count = 0

    for val_data in val_loader:
        count += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        hr_img = tensor2img(visuals['HR'])

        # SR may be a grid (continuous) — take last frame
        sr_tensor = visuals['SR']
        if sr_tensor.dim() == 4:
            sr_final = tensor2img(sr_tensor[-1])
        else:
            sr_final = tensor2img(sr_tensor)

        # Determine filename from data path
        paths = diffusion.data.get('path', ['unknown'])
        if isinstance(paths, (list, tuple)):
            path_str = paths[0] if paths else 'unknown'
        else:
            path_str = str(paths)
        filename = os.path.basename(os.path.dirname(path_str))
        if not filename or filename == '.':
            filename = f'sample_{count}'

        save_img(sr_final, os.path.join(result_path, f'{filename}_sr.png'))
        save_img(hr_img, os.path.join(result_path, f'{filename}_hr.png'))
        for i in range(1, 6):
            key = f'LR{i}'
            if key in visuals:
                save_img(
                    tensor2img(visuals[key]),
                    os.path.join(result_path, f'{filename}_lr{i}.png'),
                )

        psnr = calculate_psnr(sr_final, hr_img)
        ssim = calculate_ssim(sr_final, hr_img)
        avg_psnr += psnr
        avg_ssim += ssim

    if count > 0:
        avg_psnr /= count
        avg_ssim /= count

    logger.info(f'# Evaluation # PSNR: {avg_psnr:.4f}  SSIM: {avg_ssim:.4f}')
    val_logger = logging.getLogger('val')
    val_logger.info(f'PSNR: {avg_psnr:.4e}  SSIM: {avg_ssim:.4e}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print('=' * 60)
    print('LP-Diff Super-Resolution')
    print('=' * 60)

    cfg = build_config(args)

    if cfg.phase == 'train':
        train(cfg, args)
    else:
        evaluate(cfg, args)


if __name__ == '__main__':
    main()