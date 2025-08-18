# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import time
import csv
import math
import sys
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Scientific & Visualization Libraries
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 PyTorch Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Image Transformations & Utilities
# ───────────────────────────────────────────────────────────────────────────────
import torchvision.transforms.functional as functional_transforms
from tqdm import tqdm


# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
# Add custom project folder to system path to enable local module imports
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils.utils import default_conv 




# ───────────────────────────────────────────────────────────────────────────────
# 🎯 Mean Shift Layer
# ───────────────────────────────────────────────────────────────────────────────
class MeanShift(nn.Conv2d):
    """
    Normalizes or de-normalizes the input RGB image based on provided mean and std.
    Usually used to center input data (subtract mean).
    """
    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040), 
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

# ───────────────────────────────────────────────────────────────────────────────
# 🔁 Basic Convolutional Block
# ───────────────────────────────────────────────────────────────────────────────
class BasicBlock(nn.Sequential):
    """
    A basic convolutional block with optional BatchNorm and activation.
    """
    def __init__(self, conv, in_channels, out_channels, kernel_size, 
                 stride=1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

# ───────────────────────────────────────────────────────────────────────────────
# 🔁 Residual Block
# ───────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    Residual Block with two convolutional layers and optional BatchNorm/activation.
    Scales the residual before adding to the input.
    """
    def __init__(self, conv, n_feats, kernel_size,
                 bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

# ───────────────────────────────────────────────────────────────────────────────
# ⬆️ Upsampling Block
# ───────────────────────────────────────────────────────────────────────────────
class Upsampler(nn.Sequential):
    """
    Upsamples feature maps by scale 2^n using PixelShuffle.
    Optionally applies BatchNorm and activation after each upsample.
    """
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # If scale is power of 2
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3,bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Enhanced Deep Super-Resolution Network (EDSR)
# ───────────────────────────────────────────────────────────────────────────────
class RR_MS(nn.Module):
    """
    EDSR: A deep CNN architecture for image super-resolution.
    Consists of residual blocks, mean shift, and upsampling layers.
    """
    def __init__(self, args, conv=default_conv):
        super(RR_MS, self).__init__()

        # Parameters
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale
        act = nn.ReLU(True)

        # ───────────────────────────────
        # Band Fusion Module
        # ───────────────────────────────

        # Head: initial feature extraction
        m_head_10m = [conv(4, n_feats, kernel_size)]
        m_head_20m = [conv(4, n_feats, kernel_size)]

        # Body: series of residual blocks
        m_body_10m = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]

        m_body_10m.append(conv(n_feats, n_feats, kernel_size))

        m_body_20m = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]

        m_body_20m.append(conv(n_feats, n_feats, kernel_size))

        # ────────────────────────────────
        # Cross-Resolution Fusion Module
        # ────────────────────────────────

        fusion_upsampler = Upsampler(conv, scale, n_feats, act=False),conv(n_feats, args.n_colors, kernel_size)
        fusion_conv = conv(2*n_feats, 2*n_feats, kernel_size)


        # ────────────────────────────────────
        # Superresolution Processing Module
        # ────────────────────────────────────
        sres_module = [
            ResBlock(conv, 2*n_feats, kernel_size, act=act, res_scale=args.res_scale)
            for _ in range(n_resblocks)
        ]

        # ────────────────────────────────────
        # Upscaling Module
        # ────────────────────────────────────
        
        # Tail: upsampling and reconstruction
        up_module = [
            Upsampler(conv, scale, 2*n_feats, act=False),
            conv(2*n_feats, 8, kernel_size)
        ]

        self.head_b2b3b4b8 = nn.Sequential(m_head_10m)
        self.head_b5b6b7b8a = nn.Sequential(m_head_20m)
        
        self.bfm_b2b3b4b8 = nn.Sequential(*m_body_10m)
        self.bfm_b5b6b7b8a = nn.Sequential(*m_body_20m)

        self.crfm_upsampler_b5b6b7b8a = fusion_upsampler
        self.crfm_conv = fusion_conv

        self.srm = nn.Sequential(*sres_module)
        self.um = nn.Sequential(*up_module)

    def forward(self, x_b2b3b4b8, x_b5b6b7b8a):

        # Band Fusion Module
        x_b2b3b4b8 = self.head_b2b3b4b8(x_b2b3b4b8)
        x_b5b6b7b8a = self.head_b5b6b7b8a(x_b5b6b7b8a) 

        res_b2b3b4b8 = self.bfm_b2b3b4b8(x_b2b3b4b8)
        res_b5b6b7b8a = self.bfm_b5b6b7b8a(x_b5b6b7b8a)

        res_b2b3b4b8 += x_b2b3b4b8
        res_b5b6b7b8a += x_b5b6b7b8a

        # Cross-Resolution Fusion Module
        res_b5b6b7b8a = self.crfm_upsampler_b5b6b7b8a(res_b5b6b7b8a)
        x = torch.cat((res_b2b3b4b8, res_b5b6b7b8a), dim=1)
        x = self.crfm_conv(x)


        # Superresolution Module
        res = self.srm(x)
        res += x  # Residual connection

        # Upscaling Module
        x = self.um(res)
        return x

    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.total_params, self.trainable_params


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Enhanced Deep Super-Resolution Network CONFIG (EDSR CONFIG)
# ───────────────────────────────────────────────────────────────────────────────

class RR_MSConfig:
    def __init__(self, n_resblocks, n_feats, scale, n_colors, res_scale, rgb_range):
        self.n_resblocks = n_resblocks      # Número de bloques residuales
        self.n_feats = n_feats              # Número de características (features)
        self.scale = scale                # Escala de superresolución (lista con un elemento)
        self.n_colors = n_colors            # Número de canales (e.g. 3 para RGB)
        self.res_scale = res_scale          # Factor de escala residual
        self.rgb_range = rgb_range          # Rango de valores RGB (e.g. 255)
        print(scale)

    def __repr__(self):
        return (f"EDSRConfig(n_resblocks={self.n_resblocks}, n_feats={self.n_feats}, "
                f"scale={self.scale}, n_colors={self.n_colors}, "
                f"res_scale={self.res_scale}, rgb_range={self.rgb_range})")