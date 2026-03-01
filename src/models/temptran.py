"""AlignedResNetTransOCR: Pairwise-aligned ResNet + Temporal Transformer Fusion + Transformer head.

This model differs from ResTranOCR in three key ways:
  1. Pairwise STN alignment (aligns each frame to a center reference) instead of per-frame STN.
  2. ResNet-LR backbone that preserves spatial resolution for low-res inputs.
  3. Temporal Transformer Fusion that uses self-attention across frames at each spatial position.

Pipeline: Input (5 frames) -> [Pairwise STN Alignment] -> ResNetLR -> TemporalTransformerFusion
          -> Height Collapse -> Positional Encoding -> Transformer Encoder -> CTC Head
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import ResNetLR, STNAlignment, TemporalTransformerFusion, PositionalEncoding


class AlignedResNetTransOCR(nn.Module):
    """Multi-frame OCR with pairwise alignment, temporal transformer fusion, and CTC decoding."""

    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
    ):
        super().__init__()
        self.use_stn = use_stn
        self.cnn_channels = 512

        # 1. Pairwise Spatial Transformer alignment
        if self.use_stn:
            self.aligner = STNAlignment()

        # 2. Backbone: ResNet34 modified for low-resolution images
        self.backbone = ResNetLR()

        # 3. Temporal Transformer Fusion across frames
        self.fusion = TemporalTransformerFusion(channels=self.cnn_channels)

        # 4. Transformer Encoder for sequence modelling
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # 5. CTC Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    # ------------------------------------------------------------------
    # Alignment helpers
    # ------------------------------------------------------------------

    def _align_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Align all frames to the center frame using pairwise STN.

        Args:
            x: [B, T, C, H, W]
        Returns:
            Aligned tensor [B, T, C, H, W]
        """
        b, t, c, h, w = x.size()
        center_idx = t // 2
        center_frame = x[:, center_idx]

        aligned = []
        for i in range(t):
            if i == center_idx:
                aligned.append(center_frame)
            else:
                aligned.append(self.aligner(center_frame, x[:, i]))

        return torch.stack(aligned, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Log-softmax logits: [Batch, Seq_Len, Num_Classes]
        """
        b, t, c, h, w = x.size()

        # 1. Alignment (optional)
        if self.use_stn:
            x = self._align_frames(x)  # [B, T, C, H, W]

        # 2. Feature extraction (shared backbone over all frames)
        x_flat = x.view(b * t, c, h, w)
        feats = self.backbone(x_flat)  # [B*T, 512, H', W']

        # 3. Temporal Transformer Fusion
        _, c_f, h_f, w_f = feats.size()
        feats = feats.view(b, t, c_f, h_f, w_f)
        fused = self.fusion(feats)  # [B, 512, H', W']

        # 4. Height collapse -> sequence
        fused = F.adaptive_avg_pool2d(fused, (1, None))  # [B, 512, 1, W']
        seq_input = fused.squeeze(2).permute(0, 2, 1)     # [B, W', 512]

        # 5. Transformer encoder
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input)  # [B, W', 512]

        # 6. Prediction
        out = self.head(seq_out)  # [B, W', Num_Classes]
        return out.log_softmax(2)