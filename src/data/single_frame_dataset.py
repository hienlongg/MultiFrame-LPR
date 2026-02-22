"""Single-frame HR dataset for standalone CRNN training.

Each HR image is treated as an independent sample, yielding 5× the number of
tracks.  Only HR images are loaded (no LR, no synthetic degradation).
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
    get_train_transforms,
    get_val_transforms,
    get_light_transforms,
)


class SingleFrameHRDataset(Dataset):
    """Dataset that loads individual HR images for single-frame CRNN training.

    Each track contains up to 5 HR images sharing the same label.  Every HR
    image becomes a separate (image, label) sample.

    Returns
    -------
    image : Tensor [C, H, W]
    target : Tensor  (encoded label indices, variable length)
    target_len : int
    label : str
    track_id : str
    """

    def __init__(
        self,
        root_dir: str,
        mode: str = "train",
        split_ratio: float = 0.9,
        img_height: int = 32,
        img_width: int = 128,
        char2idx: Dict[str, int] | None = None,
        val_split_file: str = "data/val_tracks.json",
        seed: int = 42,
        augmentation_level: str = "full",
        is_test: bool = False,
        full_train: bool = False,
    ):
        self.mode = mode
        self.samples: List[Dict[str, Any]] = []
        self.img_height = img_height
        self.img_width = img_width
        self.char2idx = char2idx or {}
        self.val_split_file = val_split_file
        self.seed = seed
        self.is_test = is_test
        self.full_train = full_train

        # ---- transforms ----
        if mode == "train":
            self.transform = (
                get_light_transforms(img_height, img_width)
                if augmentation_level == "light"
                else get_train_transforms(img_height, img_width)
            )
        else:
            self.transform = get_val_transforms(img_height, img_width)

        # ---- discover tracks ----
        print(f"[{mode.upper()}] Scanning: {root_dir}")
        abs_root = os.path.abspath(root_dir)
        search_path = os.path.join(abs_root, "**", "track_*")
        all_tracks = sorted(glob.glob(search_path, recursive=True))

        if not all_tracks:
            print("❌ ERROR: No data found.")
            return

        if is_test:
            self._index_test_samples(all_tracks)
        else:
            train_tracks, val_tracks = self._split_tracks(all_tracks, split_ratio)
            selected = train_tracks if mode == "train" else val_tracks
            print(f"[{mode.upper()}] {len(selected)} tracks selected.")
            self._index_samples(selected)

        print(f"-> Total: {len(self.samples)} single-frame samples.")

    # ------------------------------------------------------------------
    # Splitting (reuses same logic as MultiFrameDataset for consistency)
    # ------------------------------------------------------------------
    def _split_tracks(
        self, all_tracks: List[str], split_ratio: float
    ) -> Tuple[List[str], List[str]]:
        if self.full_train:
            print("📌 FULL TRAIN MODE: all tracks used for training.")
            return all_tracks, []

        train_tracks: List[str] = []
        val_tracks: List[str] = []

        if os.path.exists(self.val_split_file):
            print(f"📂 Loading split from '{self.val_split_file}'...")
            try:
                with open(self.val_split_file, "r") as f:
                    val_ids = set(json.load(f))
            except Exception:
                val_ids = set()

            for t in all_tracks:
                (val_tracks if os.path.basename(t) in val_ids else train_tracks).append(t)

            scenario_b_in_val = any("Scenario-B" in t for t in val_tracks)
            if not val_tracks or (not scenario_b_in_val and len(all_tracks) > 100):
                val_tracks = []

        if not val_tracks:
            print("⚠️ Creating new split (Val from Scenario-B)...")
            scenario_b = [t for t in all_tracks if "Scenario-B" in t] or all_tracks
            val_size = max(1, int(len(scenario_b) * (1 - split_ratio)))
            random.Random(self.seed).shuffle(scenario_b)
            val_tracks = scenario_b[:val_size]
            val_set = set(val_tracks)
            train_tracks = [t for t in all_tracks if t not in val_set]

            split_data = [os.path.basename(t) for t in val_tracks]
            try:
                os.makedirs(os.path.dirname(self.val_split_file), exist_ok=True)
                with open(self.val_split_file, "w") as f:
                    json.dump(split_data, f, indent=2)
            except OSError:
                pass

        return train_tracks, val_tracks

    # ------------------------------------------------------------------
    # Indexing – one sample per HR image
    # ------------------------------------------------------------------
    def _index_samples(self, tracks: List[str]) -> None:
        for track_path in tqdm(tracks, desc=f"Indexing {self.mode}"):
            json_path = os.path.join(track_path, "annotations.json")
            if not os.path.exists(json_path):
                continue
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    data = data[0]
                label = data.get(
                    "plate_text", data.get("license_plate", data.get("text", ""))
                )
                if not label:
                    continue

                track_id = os.path.basename(track_path)
                hr_files = sorted(
                    glob.glob(os.path.join(track_path, "hr-*.png"))
                    + glob.glob(os.path.join(track_path, "hr-*.jpg"))
                )

                for hr_path in hr_files:
                    self.samples.append(
                        {"path": hr_path, "label": label, "track_id": track_id}
                    )
            except Exception:
                pass

    def _index_test_samples(self, tracks: List[str]) -> None:
        for track_path in tqdm(tracks, desc="Indexing test"):
            track_id = os.path.basename(track_path)
            hr_files = sorted(
                glob.glob(os.path.join(track_path, "hr-*.png"))
                + glob.glob(os.path.join(track_path, "hr-*.jpg"))
            )
            for hr_path in hr_files:
                self.samples.append({"path": hr_path, "label": "", "track_id": track_id})

    # ------------------------------------------------------------------
    # __getitem__ / __len__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        item = self.samples[idx]
        image = cv2.imread(item["path"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]  # [C, H, W]

        label = item["label"]
        track_id = item["track_id"]

        if self.is_test:
            target = [0]
            target_len = 1
        else:
            target = [self.char2idx[c] for c in label if c in self.char2idx]
            if not target:
                target = [0]
            target_len = len(target)

        return image, torch.tensor(target, dtype=torch.long), target_len, label, track_id

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(
        batch: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...], Tuple[str, ...]]:
        images, targets, target_lengths, labels_text, track_ids = zip(*batch)
        images = torch.stack(images, 0)
        targets = torch.cat(targets)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        return images, targets, target_lengths, labels_text, track_ids
