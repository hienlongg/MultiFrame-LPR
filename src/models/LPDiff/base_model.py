import os
import torch
import torch.nn as nn

from configs.config import LPDiffConfig


class BaseModel:
    """Base class for LP-Diff models.

    Provides device placement, network description utilities, and step/epoch
    tracking.  Accepts a typed ``LPDiffConfig`` dataclass.
    """

    def __init__(self, cfg: LPDiffConfig):
        self.cfg = cfg
        self.device = torch.device(
            'cuda' if cfg.gpu_ids else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    if isinstance(x[key], list):
                        pass
                    else:
                        x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        """Get the string and total parameters of the network."""
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
