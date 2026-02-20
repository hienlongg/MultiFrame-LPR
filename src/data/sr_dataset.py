"""Super-Resolution dataset for EDSRLite standalone training.

Provides paired (LR, HR) images for single-image super-resolution training.
LR inputs are either real low-resolution images or synthetically degraded HR images.
HR targets are the original high-resolution images resized to the target resolution.
"""
import glob
import json
import os
import random
from typing import Any, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.data.transforms import (
    get_degradation_transforms,
)

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_sr_train_transforms(img_height: int, img_width: int) -> A.Compose:
    """Light augmentation for SR training (avoid destructive transforms)."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.HorizontalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_sr_val_transforms(img_height: int, img_width: int) -> A.Compose:
    """Validation transforms for SR (resize + normalize only)."""
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


class SRDataset(Dataset):
    """Dataset for single-image super-resolution training of EDSRLite.

    Each sample is a single image pair (LR input, HR target).
    - For HR source images: synthetic degradation is applied to create LR,
      and the original HR serves as the target.
    - For real LR images: the LR image is the input and the HR image
      from the same track (if available) is the target.

    This dataset is designed for standalone EDSRLite pretraining before
    integrating into the StackedSRNet pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        split_ratio: float = 0.9,
        lr_height: int = 16,
        lr_width: int = 64,
        hr_height: int = 48,
        hr_width: int = 124,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        include_synthetic: bool = True,
        full_train: bool = False,
    ):
        """
        Args:
            root_dir: Root directory containing track folders.
            mode: 'train' or 'val'.
            split_ratio: Train/val split ratio.
            lr_height: Low-resolution input height.
            lr_width: Low-resolution input width.
            hr_height: High-resolution target height.
            hr_width: High-resolution target width.
            val_split_file: Path to validation split JSON file.
            seed: Random seed for reproducible splitting.
            include_synthetic: If True, include degraded HR as synthetic LR samples.
            full_train: If True, use all tracks for training (no val split).
        """
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.lr_height = lr_height
        self.lr_width = lr_width
        self.hr_height = hr_height
        self.hr_width = hr_width
        self.seed = seed
        self.include_synthetic = include_synthetic
        self.full_train = full_train
        self.val_split_file = val_split_file

        # Transforms
        if mode == "train":
            self.lr_transform = get_sr_train_transforms(lr_height, lr_width)
            self.hr_transform = get_sr_train_transforms(hr_height, hr_width)
            self.degrade = get_degradation_transforms()
        else:
            self.lr_transform = get_sr_val_transforms(lr_height, lr_width)
            self.hr_transform = get_sr_val_transforms(hr_height, hr_width)
            self.degrade = get_degradation_transforms()  # Still needed for synthetic pairs

        print(f"[SR-{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("No tracks found.")
            return

        train_tracks, val_tracks = self._load_or_create_split(all_tracks, split_ratio)
        selected = train_tracks if mode == "train" else val_tracks
        print(f"[SR-{mode.upper()}] {len(selected)} tracks selected.")

        self._index_samples(selected)
        print(f"-> Total: {len(self.samples)} SR sample pairs.")

    def _load_or_create_split(
        self, all_tracks: List[str], split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        """Load existing split or create new one (reuses main dataset split)."""
        if self.full_train:
            return all_tracks, []

        if os.path.exists(self.val_split_file):
            try:
                with open(self.val_split_file, "r") as f:
                    val_ids = set(json.load(f))
                train = [t for t in all_tracks if os.path.basename(t) not in val_ids]
                val = [t for t in all_tracks if os.path.basename(t) in val_ids]
                if val:
                    return train, val
            except Exception:
                pass

        # Fallback: random split
        shuffled = list(all_tracks)
        random.Random(self.seed).shuffle(shuffled)
        split_idx = int(len(shuffled) * split_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def _index_samples(self, tracks: List[str]) -> None:
        """Index individual image pairs from tracks."""
        for track_path in tqdm(tracks, desc=f"Indexing SR {self.mode}"):
            lr_files = sorted(
                glob.glob(os.path.join(track_path, "lr-*.png"))
                + glob.glob(os.path.join(track_path, "lr-*.jpg"))
            )
            hr_files = sorted(
                glob.glob(os.path.join(track_path, "hr-*.png"))
                + glob.glob(os.path.join(track_path, "hr-*.jpg"))
            )
            track_id = os.path.basename(track_path)

            # Real LR -> HR pairs (if both exist and counts match by index)
            hr_lookup = {}
            for hr_path in hr_files:
                # Extract frame index: hr-0.png -> 0
                fname = os.path.splitext(os.path.basename(hr_path))[0]
                idx = fname.split("-")[-1]
                hr_lookup[idx] = hr_path

            for lr_path in lr_files:
                fname = os.path.splitext(os.path.basename(lr_path))[0]
                idx = fname.split("-")[-1]
                hr_path = hr_lookup.get(idx)

                if hr_path is not None:
                    # Paired LR-HR sample
                    self.samples.append({
                        "lr_path": lr_path,
                        "hr_path": hr_path,
                        "is_synthetic": False,
                        "track_id": track_id,
                    })

            # Synthetic LR from HR images (only during training)
            if self.include_synthetic and self.mode == "train":
                for hr_path in hr_files:
                    self.samples.append({
                        "lr_path": None,  # Will be created via degradation
                        "hr_path": hr_path,
                        "is_synthetic": True,
                        "track_id": track_id,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return (lr_image, hr_image, track_id).

        For synthetic samples: HR is degraded to create LR.
        For real samples: LR and HR are loaded from disk.
        """
        item = self.samples[idx]
        track_id = item["track_id"]

        # Load HR image (always available)
        hr_img = cv2.imread(item["hr_path"], cv2.IMREAD_COLOR)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        if item["is_synthetic"]:
            # Create synthetic LR from HR via degradation
            lr_img = self.degrade(image=hr_img.copy())["image"]
        else:
            lr_img = cv2.imread(item["lr_path"], cv2.IMREAD_COLOR)
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        lr_tensor = self.lr_transform(image=lr_img)["image"]
        hr_tensor = self.hr_transform(image=hr_img)["image"]

        return lr_tensor, hr_tensor, track_id

    @staticmethod
    def collate_fn(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[str, ...]]:
        """Custom collate function for SR DataLoader."""
        lr_images, hr_images, track_ids = zip(*batch)
        lr_batch = torch.stack(lr_images, dim=0)
        hr_batch = torch.stack(hr_images, dim=0)
        return lr_batch, hr_batch, track_ids
