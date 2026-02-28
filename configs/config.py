"""Configuration dataclass for the training pipeline."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


# ---------------------------------------------------------------------------
# LP-Diff (diffusion-based super-resolution) sub-configs
# ---------------------------------------------------------------------------

@dataclass
class BetaScheduleConfig:
    """Noise schedule parameters for diffusion forward/reverse process."""
    schedule: str = "linear"        # "linear", "cosine", "quad", "warmup10", etc.
    n_timestep: int = 1000
    linear_start: float = 1e-6
    linear_end: float = 1e-2


@dataclass
class UNetConfig:
    """UNet denoiser architecture settings."""
    in_channel: int = 6             # concat(condition, noisy) → 3+3
    out_channel: int = 3
    inner_channel: int = 64
    channel_multiplier: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 8])
    attn_res: List[int] = field(default_factory=lambda: [16])
    res_blocks: int = 2
    dropout: float = 0.1
    norm_groups: int = 32


@dataclass
class DiffusionModelConfig:
    """High-level diffusion model settings (image size, channels, mode)."""
    image_size: int = 128
    channels: int = 3               # number of output/sample channels
    conditional: bool = True        # conditional (SR) vs unconditional generation
    loss_type: str = "l1"           # "l1" or "l2"


@dataclass
class EMAConfig:
    """Exponential moving average scheduler for diffusion training."""
    step_start_ema: int = 5000
    update_ema_every: int = 1
    ema_decay: float = 0.9999


@dataclass
class LPDiffOptimizerConfig:
    """Optimizer settings for LP-Diff training."""
    type: str = "adam"
    lr: float = 1e-4


@dataclass
class LPDiffTrainConfig:
    """LP-Diff training loop parameters."""
    use_pretrain_mta: bool = False          # load pretrained MTA weights
    resume_training: bool = False           # resume from checkpoint
    mta_checkpoint: str = "./best_377.pt"   # path to pretrained MTA weights
    epochs: int = 100                       # total training epochs
    val_freq: int = 5                       # validate every N epochs
    save_checkpoint_freq: int = 10          # save checkpoint every N epochs
    print_freq: int = 200                   # log every N iterations
    optimizer: LPDiffOptimizerConfig = field(default_factory=LPDiffOptimizerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)


@dataclass
class LPDiffDatasetConfig:
    """Dataset settings for a single LP-Diff data split."""
    name: str = "MDLP"
    mode: str = "LRHR"
    dataroot: str = "data/train"
    width: int = 224
    height: int = 112
    batch_size: int = 20
    num_workers: int = 0


@dataclass
class LPDiffPathConfig:
    """Directory layout for LP-Diff experiment outputs."""
    log: str = "logs"
    tb_logger: str = "tb_logger"
    results: str = "results"
    checkpoint: str = "checkpoint"
    resume_state: Optional[str] = None    # e.g. "./I1000000_E3047"


@dataclass
class LPDiffConfig:
    """Top-level configuration for the LP-Diff diffusion SR model.

    Mirrors the original JSON config while using typed dataclasses for
    validation, IDE autocompletion, and consistency with the rest of the
    codebase.
    """
    # Phase & device
    phase: str = "train"                    # "train" or "val"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    distributed: bool = False

    # Paths
    path: LPDiffPathConfig = field(default_factory=LPDiffPathConfig)

    # Datasets
    train_dataset: LPDiffDatasetConfig = field(default_factory=LPDiffDatasetConfig)
    val_dataset: LPDiffDatasetConfig = field(
        default_factory=lambda: LPDiffDatasetConfig(batch_size=1, num_workers=0)
    )

    # Model architecture
    finetune_norm: bool = False
    unet: UNetConfig = field(default_factory=UNetConfig)
    beta_schedule_train: BetaScheduleConfig = field(default_factory=BetaScheduleConfig)
    beta_schedule_val: BetaScheduleConfig = field(default_factory=BetaScheduleConfig)
    diffusion: DiffusionModelConfig = field(default_factory=DiffusionModelConfig)

    # Training
    train: LPDiffTrainConfig = field(default_factory=LPDiffTrainConfig)

    # Logging
    wandb_project: str = "LP-Diff"

    def to_dict(self) -> dict:
        """Convert to the nested dict format expected by the LPDiff model code.

        This bridges the typed dataclass config with the dict-based interface
        used by ``networks.define_G(opt)`` and ``DDPM(opt)``."""
        return {
            "name": "LP-Diff",
            "phase": self.phase,
            "gpu_ids": self.gpu_ids,
            "distributed": self.distributed,
            "path": {
                "log": self.path.log,
                "tb_logger": self.path.tb_logger,
                "results": self.path.results,
                "checkpoint": self.path.checkpoint,
                "resume_state": self.path.resume_state,
            },
            "datasets": {
                "train": {
                    "name": self.train_dataset.name,
                    "mode": self.train_dataset.mode,
                    "dataroot": self.train_dataset.dataroot,
                    "width": self.train_dataset.width,
                    "height": self.train_dataset.height,
                    "batch_size": self.train_dataset.batch_size,
                    "num_workers": self.train_dataset.num_workers,
                },
                "val": {
                    "name": self.val_dataset.name,
                    "mode": self.val_dataset.mode,
                    "dataroot": self.val_dataset.dataroot,
                    "width": self.val_dataset.width,
                    "height": self.val_dataset.height,
                },
            },
            "model": {
                "finetune_norm": self.finetune_norm,
                "unet": {
                    "in_channel": self.unet.in_channel,
                    "out_channel": self.unet.out_channel,
                    "inner_channel": self.unet.inner_channel,
                    "channel_multiplier": self.unet.channel_multiplier,
                    "attn_res": self.unet.attn_res,
                    "res_blocks": self.unet.res_blocks,
                    "dropout": self.unet.dropout,
                    "norm_groups": self.unet.norm_groups,
                },
                "beta_schedule": {
                    "train": {
                        "schedule": self.beta_schedule_train.schedule,
                        "n_timestep": self.beta_schedule_train.n_timestep,
                        "linear_start": self.beta_schedule_train.linear_start,
                        "linear_end": self.beta_schedule_train.linear_end,
                    },
                    "val": {
                        "schedule": self.beta_schedule_val.schedule,
                        "n_timestep": self.beta_schedule_val.n_timestep,
                        "linear_start": self.beta_schedule_val.linear_start,
                        "linear_end": self.beta_schedule_val.linear_end,
                    },
                },
                "diffusion": {
                    "image_size": self.diffusion.image_size,
                    "channels": self.diffusion.channels,
                    "conditional": self.diffusion.conditional,
                },
            },
            "train": {
                "use_prerain_MTA": self.train.use_pretrain_mta,
                "resume_training": self.train.resume_training,
                "MTA": self.train.mta_checkpoint,
                "n_iter": self.train.n_iter,
                "val_freq": self.train.val_freq,
                "save_checkpoint_freq": self.train.save_checkpoint_freq,
                "print_freq": self.train.print_freq,
                "optimizer": {
                    "type": self.train.optimizer.type,
                    "lr": self.train.optimizer.lr,
                },
                "ema_scheduler": {
                    "step_start_ema": self.train.ema.step_start_ema,
                    "update_ema_every": self.train.ema.update_ema_every,
                    "ema_decay": self.train.ema.ema_decay,
                },
            },
            "wandb": {"project": self.wandb_project},
        }


# ---------------------------------------------------------------------------
# Main OCR pipeline config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    MODEL_TYPE: str = "restran"  # "crnn" or "restran"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network
    
    # Data paths
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"
    
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 30
    SEED: int = 42
    NUM_WORKERS: int = 10
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False
    
    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    
    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.1
    
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"
    
    # LP-Diff super-resolution config (optional, only used when MODEL_TYPE involves diffusion SR)
    LPDIFF: LPDiffConfig = field(default_factory=LPDiffConfig)
    
    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()


def get_lpdiff_config(**overrides) -> LPDiffConfig:
    """Returns LP-Diff configuration with optional overrides.

    Example:
        cfg = get_lpdiff_config(phase="val", gpu_ids=[0, 1])
    """
    return LPDiffConfig(**overrides)
