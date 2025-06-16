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
# 🎯 Channel Attention Module
# ───────────────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism.
    Uses global average pooling followed by a bottleneck (conv-relu-conv-sigmoid)
    to learn channel-wise dependencies.
    """
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)  # Scale input features channel-wise


# ───────────────────────────────────────────────────────────────────────────────
# 🧱 Residual Channel Attention Block (RCAB)
# ───────────────────────────────────────────────────────────────────────────────
class RCAB(nn.Module):
    """
    Residual Channel Attention Block.
    Applies two convolutional layers with ReLU and integrates channel attention.
    Includes a residual connection.
    """
    def __init__(self, num_features, reduction):
        super(RCAB, self).__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x + self.module(x)  # Residual connection


# ───────────────────────────────────────────────────────────────────────────────
# 🧱 Residual Group (RG)
# ───────────────────────────────────────────────────────────────────────────────
class RG(nn.Module):
    """
    Residual Group.
    Consists of a sequence of RCABs followed by a convolutional layer,
    with an overall residual connection.
    """
    def __init__(self, num_features, num_rcab, reduction):
        super(RG, self).__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x + self.module(x)  # Residual connection over the whole group


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 RCAN Network
# ───────────────────────────────────────────────────────────────────────────────
class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN).
    Deep network for image super-resolution using residual-in-residual structure
    with channel attention and pixel shuffle upsampling.
    """
    def __init__(self, args):
        super(RCAN, self).__init__()
        scale = args.scale
        num_features = args.num_features
        num_rg = args.num_rg
        num_rcab = args.num_rcab
        reduction = args.reduction

        # Initial feature extraction
        self.sf = default_conv(in_channels=3,out_channels=num_features,kernel_size=3)

        # Residual groups (each contains multiple RCABs)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction) for _ in range(num_rg)])

        # Convolution after residual groups
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # Upsampling module using PixelShuffle
        self.upscale = nn.Sequential(
            nn.Conv2d(num_features, num_features * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

        # Final convolution to get RGB output
        self.conv2 = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual  # Global residual connection
        x = self.upscale(x)
        x = self.conv2(x)
        return x
    
    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.total_params, self.trainable_params
