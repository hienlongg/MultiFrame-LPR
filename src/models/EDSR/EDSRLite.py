import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv, ResBlock, Upsampler

class EDSRLite(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, 
                 num_features=64, num_blocks=16,
                 res_scale=0.1, target_size=(43, 121)):
        super(EDSRLite, self).__init__()
        self.target_size = target_size
        out_channels = out_channels if out_channels is not None else in_channels
        
        # Head: Initial feature extraction
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Body: Residual blocks
        body = [ResBlock(num_features, res_scale) for _ in range(num_blocks)]
        body.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)
        
        # Tail: Output projection
        tail = [Upsampler(conv, scale=2, n_feats=num_features, act=False), 
                conv(num_features, out_channels, kernel_size=3)]
        self.tail = nn.Sequential(*tail)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor resized to target_size (B, out_channels, target_H, target_W)
        """
        # Feature extraction
        x = self.head(x)
        
        # Residual feature processing with skip connection
        res = self.body(x)
        res += x
        
        # Output projection
        x = self.tail(res)
        
        # Resize to target size if specified
        if self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        
        return x
        