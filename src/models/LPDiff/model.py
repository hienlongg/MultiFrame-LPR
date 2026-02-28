"""DDPM wrapper for LP-Diff training, validation, and inference.

Orchestrates the GaussianDiffusion network (UNet + MTA) with optimizer
setup, checkpointing, and noise schedule management.  Accepts a typed
``LPDiffConfig`` dataclass from ``configs.config``.
"""

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from configs.config import LPDiffConfig
from . import networks
from .base_model import BaseModel

logger = logging.getLogger('base')

# Keys for the 5 LR frames expected by MTA
_LR_KEYS = ['LR1', 'LR2', 'LR3', 'LR4', 'LR5']


class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model for license-plate SR."""

    def __init__(self, cfg: LPDiffConfig):
        super().__init__(cfg)
        # Build network
        self.netG = self.set_device(networks.define_G(cfg))
        self.schedule_phase = None

        # Loss & noise schedule
        self.set_loss()
        self.set_new_noise_schedule(
            cfg.beta_schedule_train, schedule_phase='train')

        if cfg.phase == 'train':
            self.netG.train()
            # Optionally freeze everything except transformer layers
            if cfg.finetune_norm:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if 'transformer' in k:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=cfg.train.optimizer.lr)
            self.log_dict = OrderedDict()

            if cfg.train.resume_training:
                self.load_network()
            if cfg.train.use_pretrain_mta:
                checkpoint = torch.load(
                    cfg.train.mta_checkpoint, map_location=self.device)
                self.netG.MTA.load_state_dict(
                    checkpoint['model_state_dict'])
                logger.info('Loaded pretrained MTA model successfully.')
        else:
            self.load_network()

    # ------------------------------------------------------------------
    # Data & optimisation
    # ------------------------------------------------------------------

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # Average pixel loss over spatial dimensions
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        # Clip gradients to prevent explosion (common for diffusion models)
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        self.optG.step()
        self.log_dict['l_pix'] = l_pix.item()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _get_mta_condition(self):
        """Fuse all 5 LR frames through MTA."""
        mta = self.netG.module.MTA if isinstance(
            self.netG, nn.DataParallel) else self.netG.MTA
        return mta(*(self.data[k] for k in _LR_KEYS))

    def test(self, continous=False):
        """Run super-resolution inference using all 5 LR frames."""
        self.netG.eval()
        with torch.no_grad():
            condition = self._get_mta_condition()
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    condition, continous)
            else:
                self.SR = self.netG.super_resolution(condition, continous)
        mse_loss = nn.MSELoss()
        loss = mse_loss(self.SR, self.data['HR'])
        self.netG.train()
        return loss

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    # ------------------------------------------------------------------
    # Loss & schedule
    # ------------------------------------------------------------------

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_cfg, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_cfg, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_cfg, self.device)

    # ------------------------------------------------------------------
    # Logging & visualisation
    # ------------------------------------------------------------------

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            for key in _LR_KEYS:
                out_dict[key] = self.data[key].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netG.__class__.__name__,
                self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(
                net_struc_str, n))
        logger.info(s)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch, iter_step, suffix=''):
        """Save generator weights and optimizer state.

        Args:
            epoch: Current epoch number.
            iter_step: Current iteration number.
            suffix: Optional filename suffix (e.g. ``'_best_loss'``).
        """
        ckpt_dir = self.cfg.path.checkpoint
        gen_path = os.path.join(
            ckpt_dir, 'I{}_E{}_gen{}.pth'.format(iter_step, epoch, suffix))
        opt_path = os.path.join(
            ckpt_dir, 'I{}_E{}_opt{}.pth'.format(iter_step, epoch, suffix))
        # Generator state
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # Optimizer state
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'scheduler': None,
            'optimizer': self.optG.state_dict(),
        }
        torch.save(opt_state, opt_path)
        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def save_network(self, epoch, iter_step):
        self._save_checkpoint(epoch, iter_step)

    def save_best_loss(self, epoch, iter_step):
        self._save_checkpoint(epoch, iter_step, suffix='_best_loss')

    def save_best_psnr(self, epoch, iter_step):
        self._save_checkpoint(epoch, iter_step, suffix='_best_psnr')

    def save_best_both(self, epoch, iter_step):
        self._save_checkpoint(epoch, iter_step, suffix='_best')

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def load_network(self):
        load_path = self.cfg.path.resume_state
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # Generator
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(
                torch.load(gen_path, map_location=self.device),
                strict=(not self.cfg.finetune_norm))
            if self.cfg.phase == 'train':
                # Optimizer
                opt = torch.load(opt_path, map_location=self.device)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
