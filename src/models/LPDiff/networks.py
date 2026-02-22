import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

from configs.config import LPDiffConfig

logger = logging.getLogger('base')


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    """Apply weight initialization to the network."""
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


def define_G(cfg: LPDiffConfig):
    """Build the generator (UNet + GaussianDiffusion) from a typed config.

    Args:
        cfg: ``LPDiffConfig`` dataclass with ``unet``, ``diffusion``, and
             ``beta_schedule_train`` sub-configs.
    """
    from .LPDiff_modules import diffusion, unet

    model = unet.UNet(
        in_channel=cfg.unet.in_channel,
        out_channel=cfg.unet.out_channel,
        norm_groups=cfg.unet.norm_groups,
        inner_channel=cfg.unet.inner_channel,
        channel_mults=cfg.unet.channel_multiplier,
        attn_res=cfg.unet.attn_res,
        res_blocks=cfg.unet.res_blocks,
        dropout=cfg.unet.dropout,
        image_size=cfg.diffusion.image_size,
    )
    netG = diffusion.GaussianDiffusion(
        model,
        image_size=cfg.diffusion.image_size,
        channels=cfg.diffusion.channels,
        loss_type=cfg.diffusion.loss_type,
        conditional=cfg.diffusion.conditional,
    )
    if cfg.phase == 'train':
        init_weights(netG, init_type='orthogonal')
    if cfg.gpu_ids and cfg.distributed:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
