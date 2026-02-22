"""Trainer for the standalone single-frame CRNN model.

Mirrors the interface of ``src.training.trainer.Trainer`` but works with
the single-image CRNN (output shape ``[seq_len, batch, num_class]``) and
applies ``log_softmax`` before CTC loss.
"""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything


class CRNNTrainer:
    """Training loop for the standalone CRNN (single-frame CTC model)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE

        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)

        # CTC loss expects log-probabilities
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS,
        )
        self.scaler = GradScaler()

        self.best_acc = 0.0
        self.current_epoch = 0

    # ----- helpers -----
    def _output_path(self, filename: str) -> str:
        out = getattr(self.config, "OUTPUT_DIR", "results")
        os.makedirs(out, exist_ok=True)
        return os.path.join(out, filename)

    def _exp_name(self) -> str:
        return getattr(self.config, "EXPERIMENT_NAME", "crnn_standalone")

    # ----- greedy CTC decode -----
    @staticmethod
    def _greedy_decode(
        log_probs: torch.Tensor, idx2char: Dict[int, str]
    ) -> List[Tuple[str, float]]:
        """Greedy CTC decoding with confidence.

        Args:
            log_probs: [seq_len, batch, num_class] (log-softmax output).
            idx2char: index → char mapping (blank = 0 is excluded).

        Returns:
            List of (decoded_string, confidence) per batch element.
        """
        # transpose to [batch, seq_len, num_class]
        log_probs = log_probs.permute(1, 0, 2)
        probs = log_probs.exp()
        max_probs, indices = probs.max(dim=2)
        indices_np = indices.detach().cpu().numpy()
        max_probs_np = max_probs.detach().cpu().numpy()

        results: List[Tuple[str, float]] = []
        for b in range(indices_np.shape[0]):
            chars, confs = [], []
            prev = -1
            for t in range(indices_np.shape[1]):
                c = int(indices_np[b, t])
                if c != 0 and c != prev:
                    chars.append(idx2char.get(c, ""))
                    confs.append(float(max_probs_np[b, t]))
                prev = c
            text = "".join(chars)
            conf = float(sum(confs) / len(confs)) if confs else 0.0
            results.append((text, conf))
        return results

    # ----- train one epoch -----
    def train_one_epoch(self) -> float:
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            self.train_loader,
            desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}",
        )

        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda"):
                # CRNN output: [seq_len, batch, num_class] (raw logits)
                logits = self.model(images)
                log_probs = F.log_softmax(logits, dim=2)

                # CTC expects [T, B, C] log-probs
                input_lengths = torch.full(
                    (images.size(0),), logits.size(0), dtype=torch.long
                )
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP
            )

            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"}
            )

        return epoch_loss / len(self.train_loader)

    # ----- validate -----
    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        if self.val_loader is None:
            return {"loss": 0.0, "acc": 0.0}, []

        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        submission_data: List[str] = []

        with torch.no_grad():
            for images, targets, target_lengths, labels_text, track_ids in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                log_probs = F.log_softmax(logits, dim=2)

                input_lengths = torch.full(
                    (images.size(0),), logits.size(0), dtype=torch.long
                )
                loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

                decoded = self._greedy_decode(log_probs, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded):
                    gt = labels_text[i]
                    if pred_text == gt:
                        total_correct += 1
                    submission_data.append(f"{track_ids[i]},{pred_text};{conf:.4f}")
                total_samples += len(labels_text)

        avg_loss = val_loss / len(self.val_loader)
        acc = (total_correct / total_samples * 100) if total_samples else 0.0
        return {"loss": avg_loss, "acc": acc}, submission_data

    # ----- save -----
    def save_model(self, path: str | None = None) -> None:
        if path is None:
            path = self._output_path(f"{self._exp_name()}_best.pth")
        torch.save(self.model.state_dict(), path)

    def save_submission(self, data: List[str]) -> None:
        path = self._output_path(f"submission_{self._exp_name()}.txt")
        with open(path, "w") as f:
            f.write("\n".join(data))
        print(f"📝 Saved {len(data)} lines → {path}")

    # ----- full training loop -----
    def fit(self) -> None:
        print(
            f"🚀 CRNN TRAINING | Device: {self.device} | "
            f"Epochs: {self.config.EPOCHS}"
        )

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            train_loss = self.train_one_epoch()
            val_metrics, sub_data = self.validate()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {lr:.2e}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                print(
                    f"  ⭐ Best model saved ({val_acc:.2f}%) → "
                    f"{self._output_path(self._exp_name() + '_best.pth')}"
                )
                if sub_data:
                    self.save_submission(sub_data)

        if self.val_loader is None:
            self.save_model()
            print(f"  💾 Final model saved → {self._output_path(self._exp_name() + '_best.pth')}")

        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")
