"""LP-Diff SR dataset: loads multi-frame LR/HR track data for diffusion training.

Each track folder contains ``lr-001..lr-005.{png,jpg}`` (low-resolution) and
``hr-001..hr-005.{png,jpg}`` (high-resolution) images.  The dataset returns a
dict ``{LR1..LR5, HR, path}`` ready for ``DDPM.feed_data()``.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from configs.config import LPDiffDatasetConfig
from src.data.lpdiff_transforms import (
    get_lpdiff_train_transforms,
    get_lpdiff_val_transforms,
    apply_transform,
)

logger = logging.getLogger('base')

NUM_LR_FRAMES = 5


class LPDiffDataset(Dataset):
    """Multi-frame LR/HR dataset for LP-Diff diffusion SR training.

    Args:
        dataset_cfg: ``LPDiffDatasetConfig`` with dataroot, height, width, etc.
        phase: ``'train'`` or ``'val'``.
        val_split_file: Path to JSON file listing validation track folder names.
        split_ratio: Fraction of data used for training (rest goes to val).
        seed: Random seed for reproducible splitting.
    """

    def __init__(
        self,
        dataset_cfg: LPDiffDatasetConfig,
        phase: str = 'train',
        val_split_file: str = 'data/val_tracks.json',
        split_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()
        self.phase = phase
        self.height = dataset_cfg.height
        self.width = dataset_cfg.width
        self.seed = seed

        # Build transforms
        if phase == 'train':
            self.transform = get_lpdiff_train_transforms(self.height, self.width)
        else:
            self.transform = get_lpdiff_val_transforms(self.height, self.width)

        self.to_tensor = T.ToTensor()

        # Index samples: list of (lr_paths_5, hr_path) tuples
        self.samples: List[Tuple[List[str], str]] = []
        self._index_tracks(dataset_cfg.dataroot, phase, val_split_file, split_ratio)

        logger.info(
            f'[LP-Diff {phase}] Created dataset with {len(self.samples)} samples '
            f'(H={self.height}, W={self.width})'
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _index_tracks(
        self,
        dataroot: str,
        phase: str,
        val_split_file: str,
        split_ratio: float,
    ) -> None:
        """Discover all track folders and split into train/val."""
        abs_root = os.path.abspath(dataroot)
        all_tracks = sorted(
            glob.glob(os.path.join(abs_root, '**', 'track_*'), recursive=True)
        )
        if not all_tracks:
            logger.warning(f'No track folders found under {abs_root}')
            return

        train_tracks, val_tracks = self._load_or_create_split(
            all_tracks, val_split_file, split_ratio
        )
        selected = train_tracks if phase == 'train' else val_tracks

        for track_path in tqdm(selected, desc=f'Indexing LP-Diff {phase}'):
            lr_files = sorted(
                glob.glob(os.path.join(track_path, 'lr-*.[pj][np]g'))
            )
            hr_files = sorted(
                glob.glob(os.path.join(track_path, 'hr-*.[pj][np]g'))
            )
            if len(lr_files) < 1 or len(hr_files) < 1:
                continue
            self.samples.append((lr_files, hr_files))

    def _load_or_create_split(
        self,
        all_tracks: List[str],
        val_split_file: str,
        split_ratio: float,
    ) -> Tuple[List[str], List[str]]:
        """Load an existing train/val split or create one.

        Prioritises Scenario-B tracks for the validation set (matching
        the OCR pipeline convention).
        """
        train_tracks: List[str] = []
        val_tracks: List[str] = []

        if os.path.exists(val_split_file):
            with open(val_split_file, 'r') as f:
                val_ids = set(json.load(f))
            for t in all_tracks:
                if os.path.basename(t) in val_ids:
                    val_tracks.append(t)
                else:
                    train_tracks.append(t)
            if val_tracks:
                return train_tracks, val_tracks

        # Fallback: create random split, preferring Scenario-B for val
        scenario_b = [t for t in all_tracks if 'Scenario-B' in t]
        pool = scenario_b if scenario_b else all_tracks
        val_size = max(1, int(len(pool) * (1 - split_ratio)))
        rng = random.Random(self.seed)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        val_tracks = shuffled[:val_size]
        val_set = set(val_tracks)
        train_tracks = [t for t in all_tracks if t not in val_set]

        # Save split for reproducibility
        try:
            os.makedirs(os.path.dirname(val_split_file) or '.', exist_ok=True)
            with open(val_split_file, 'w') as f:
                json.dump([os.path.basename(t) for t in val_tracks], f, indent=2)
        except OSError:
            pass

        return train_tracks, val_tracks

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load 5 LR frames + 1 HR frame, resize, normalise, return as dict.

        During training the HR frame is chosen randomly from available HRs;
        during validation the first HR frame is always used.
        """
        lr_paths, hr_paths = self.samples[idx]

        # --- Select 5 LR frames (pad if fewer available) ---
        lr_selected = self._select_frames(lr_paths, NUM_LR_FRAMES)
        lr_images = [self._load_and_transform(p) for p in lr_selected]

        # --- Select 1 HR frame ---
        if self.phase == 'train':
            hr_path = random.choice(hr_paths)
        else:
            hr_path = hr_paths[0]
        hr_image = self._load_and_transform(hr_path)

        out: Dict[str, torch.Tensor] = {'HR': hr_image}
        for i, lr_t in enumerate(lr_images, start=1):
            out[f'LR{i}'] = lr_t
        out['path'] = hr_path  # type: ignore[assignment]
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_frames(paths: List[str], n: int) -> List[str]:
        """Return exactly *n* frame paths, repeating the last if needed."""
        if len(paths) >= n:
            return paths[:n]
        return paths + [paths[-1]] * (n - len(paths))

    def _load_and_transform(self, path: str) -> torch.Tensor:
        """Read an image, apply resize + normalise, convert to tensor."""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = apply_transform(image, self.transform)
        return self.to_tensor(image)


# ---------------------------------------------------------------------------
# Factory helpers (mirror the original LP-Diff ``Data.create_*`` interface)
# ---------------------------------------------------------------------------

def create_dataset(
    dataset_cfg: LPDiffDatasetConfig,
    phase: str,
    val_split_file: str = 'data/val_tracks.json',
    split_ratio: float = 0.9,
    seed: int = 42,
) -> LPDiffDataset:
    """Create an ``LPDiffDataset`` from a typed config."""
    return LPDiffDataset(
        dataset_cfg,
        phase=phase,
        val_split_file=val_split_file,
        split_ratio=split_ratio,
        seed=seed,
    )


def create_dataloader(
    dataset: LPDiffDataset,
    dataset_cfg: LPDiffDatasetConfig,
    phase: str,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for an ``LPDiffDataset``."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=(phase == 'train'),
        num_workers=dataset_cfg.num_workers,
        pin_memory=True,
        drop_last=(phase == 'train'),
    )
