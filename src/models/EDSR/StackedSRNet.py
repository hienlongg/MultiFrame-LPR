import torch
import torch.nn as nn
import torch.nn.functional as F
from .EDSRLite import EDSRLite
from src.models.components import STNBlock
from src.models.CRNN.model import CRNN
from src.models.CRNN.ctc_decoder import ctc_decode

class StackedSRNet(nn.Module):
    def __init__(self, in_channels=3, num_features=64, num_blocks=16, res_scale=0.1,
                 num_frames=5, target_size=(48, 124)):
        super(StackedSRNet, self).__init__()
        
        h, w = target_size
        
        self.cnn_channels = 512
        self.stn = STNBlock(in_channels=3)
        
        # Effective input channels = channels * num_frames
        stacked_channels = in_channels * num_frames  # Assuming 5 frames (current + 4 neighbors)
        
        self.sr_backbone = EDSRLite(
            in_channels=stacked_channels,
            out_channels=in_channels,
            num_features=num_features,
            num_blocks=num_blocks,
            res_scale=res_scale,
            target_size=target_size
        )
        
        self.crnn = CRNN(img_channel=in_channels, img_height=h, img_width=w, num_class=37)
        self.ctc_decoder = ctc_decode
        
    def forward(self, x):
        b, t, c, h, w = x.size()
        x_flat = x.view(b * t, c, h, w)  # [B*F, C, H, W]
        
        theta = self.stn(x_flat)  # [B*F, 2, 3]
        grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
        x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        
        x_stacked = x_aligned.view(b, t * c, h, w)  # Reshape to (B, T*C, H, W)
        hr_img = self.sr_backbone(x_stacked)  # Pass through EDSRLite
        features = self.crnn(hr_img)  # Extract features for OCR
        outputs = self.ctc_decoder(features)  # Decode with CTC
        return outputs