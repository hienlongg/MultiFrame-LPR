import torch
import torch.nn as nn
from .EDSRLite import EDSRLite

class StackedSRNet(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=16, res_scale=0.1,
                 num_frames=5, target_size=(43, 121)):
        super(StackedSRNet, self).__init__()
        
        # Effective input channels = channels * num_frames
        stacked_channels = in_channels * num_frames  # Assuming 5 frames (current + 4 neighbors)
        
        self.backbone = EDSRLite(
            in_channels=stacked_channels,
            out_channels=in_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            res_scale=res_scale,
            target_size=target_size
        )
    def forward(self, x):
        b, t, c, h, w = x.size()  # x shape: (B, T, C, H, W)
        x_stacked = x.view(b, t * c, h, w)  # Reshape to (B, T*C, H, W)
        out = self.backbone(x_stacked)  # Pass through EDSRLite
        return out