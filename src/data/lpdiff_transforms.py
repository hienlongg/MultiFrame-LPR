"""Transform pipelines for LP-Diff super-resolution training."""
import albumentations as A
import numpy as np


def get_lpdiff_train_transforms(height: int, width: int) -> A.Compose:
    """Training transforms for LP-Diff: resize + normalize to [-1, 1].

    No heavy augmentation — the diffusion model learns the residual and
    augmenting geometry would break the LR→HR correspondence.
    """
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def get_lpdiff_val_transforms(height: int, width: int) -> A.Compose:
    """Validation transforms for LP-Diff: resize + normalize to [-1, 1]."""
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])


def apply_transform(image: np.ndarray, transform: A.Compose) -> np.ndarray:
    """Apply an albumentations transform pipeline to a single image.

    Args:
        image: HxWxC uint8 numpy array (RGB).
        transform: Albumentations Compose pipeline.

    Returns:
        Transformed HxWxC float32 numpy array.
    """
    return transform(image=image)['image']
