"""Image quality metrics and visualisation helpers for LP-Diff."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Tensor ↔ image conversion
# ---------------------------------------------------------------------------

def tensor2img(tensor: torch.Tensor, out_type: type = np.uint8) -> np.ndarray:
    """Convert a [-1, 1] float tensor to a [0, 255] uint8 numpy image (HWC RGB).

    Handles single images (CxHxW) and batches / grids (NxCxHxW).
    """
    tensor = tensor.squeeze().float().detach().cpu().clamp_(-1, 1)
    # [-1, 1] → [0, 1]
    tensor = (tensor + 1) / 2
    if tensor.dim() == 3:
        img_np = tensor.permute(1, 2, 0).numpy()
    elif tensor.dim() == 4:
        # Grid: stack vertically
        imgs = [t.permute(1, 2, 0).numpy() for t in tensor]
        img_np = np.concatenate(imgs, axis=0)
    else:
        img_np = tensor.numpy()

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round().clip(0, 255).astype(np.uint8)
    return img_np


def save_img(img: np.ndarray, path: str) -> None:
    """Save a RGB uint8 numpy image to disk (converts to BGR for OpenCV)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two uint8 RGB images.

    Args:
        img1: HxWxC uint8 numpy array.
        img2: HxWxC uint8 numpy array (same shape as img1).

    Returns:
        PSNR in dB.  Returns ``float('inf')`` for identical images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two uint8 RGB images (mean over channels).

    Uses a simplified sliding-window SSIM with an 11x11 Gaussian kernel.

    Args:
        img1: HxWxC uint8 numpy array.
        img2: HxWxC uint8 numpy array (same shape as img1).

    Returns:
        Mean SSIM ∈ [0, 1].
    """
    if img1.ndim == 2:
        return _ssim_single_channel(img1, img2)
    channels = img1.shape[2]
    ssim_sum = sum(
        _ssim_single_channel(img1[:, :, c], img2[:, :, c])
        for c in range(channels)
    )
    return ssim_sum / channels


def _ssim_single_channel(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM for a single greyscale channel (HxW uint8 arrays)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())
