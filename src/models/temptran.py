import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import ResNetLR, STNAlignment, TemporalTransformerFusion, PositionalEncoding

class AlignedResNetTransOCR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.aligner = STNAlignment()
        self.cnn_channels = 512

        self.backbone = ResNetLR()
        self.fusion = TemporalTransformerFusion(channels=self.cnn_channels) # Uses your existing Fusion code
        
        # Decoder / Head (Use your preferred one: RNN or Transformer)
        
        # Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )        
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.fc = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x):
        # x: (Batch, 5, 3, H, W)
        b, t, c, h, w = x.size()
        
        # 1. ALIGNMENT
        # We pick the middle frame (index 2) as the "Anchor"
        center_frame = x[:, 2, :, :, :]
        
        aligned_frames = []
        for i in range(t):
            if i == 2:
                aligned_frames.append(center_frame)
            else:
                # Align current frame to center frame
                current_frame = x[:, i, :, :, :]
                aligned = self.aligner(center_frame, current_frame)
                aligned_frames.append(aligned)
        
        # Stack back: (Batch, 5, 3, H, W) where everything is aligned
        x_aligned = torch.stack(aligned_frames, dim=1)
        
        # 2. FEATURE EXTRACTION (Process all frames)
        x_aligned = x_aligned.view(b*t, c, h, w)
        feats = self.backbone(x_aligned) # (B*T, 512, H/8, W/8) - less downsampling now!
        
        # 3. FUSION (Now works great because features are aligned)
        _, c_f, h_f, w_f = feats.size()
        feats = feats.view(b, t, c_f, h_f, w_f)
        fused = self.fusion(feats) # (B, 512, H/8, W/8)
        
        # 4. RECOGNITION
        fused = fused.mean(dim=2) # Collapse Height
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.fc(seq_out)              # [B, W', Num_Classes]
        return out.log_softmax(2)