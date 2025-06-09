# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import time
import datetime
from pathlib import Path
from multiprocessing import Process, Queue
import csv
import math

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Scientific & Visualization Libraries
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting in background
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 PyTorch Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader, Dataset, random_split

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Image Transformations & Utilities
# ───────────────────────────────────────────────────────────────────────────────
import torchvision.transforms.functional as functional_transforms


# ───────────────────────────────────────────────────────────────────────────────
# 🛠 Utility Functions
# ───────────────────────────────────────────────────────────────────────────────

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a normalized numpy image array.
    Args:
        tensor (Tensor): Input tensor with shape [C, H, W] or [1, C, H, W].
    Returns:
        numpy.ndarray: Normalized image array with shape [H, W, C].
    """
    img = tensor.permute(1, 2, 0).cpu().numpy()  # Rearrange to [H, W, C]
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
    return img


def default_conv(in_channels, out_channels, kernel_size, padding_mode='zeros', bias=True):
    """
    Create a 2D convolutional layer with 'same' padding.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of convolution kernel.
        padding_mode (str): Padding mode, default 'zeros'.
        bias (bool): Whether to use bias term.
    Returns:
        nn.Conv2d: 2D convolutional layer.
    """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), padding_mode=padding_mode, bias=bias
    )






