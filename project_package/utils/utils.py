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
# Custom Imports
# ───────────────────────────────────────────────────────────────────────────────

from project_package.loss_functions.gradient_variance_loss import GradientVariance
from project_package.loss_functions.histogram_loss import HistogramLoss

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


def default_conv(in_channels, out_channels, kernel_size, padding_mode='replicate', bias=True):
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


def extract_patches(images: torch.Tensor, patch_size: tuple, stride: tuple) -> torch.Tensor:
    """
    Extract sliding 2D patches from a batch of multi-channel images with custom vertical and horizontal stride.

    Parameters
    ----------
    images : torch.Tensor
        Input tensor of shape (N, C, H, W), where:
        - N is the number of images,
        - C is the number of channels (e.g., 3 for RGB),
        - H is the height of each image,
        - W is the width of each image.

    patch_size : tuple of int
        The size of the patches to extract (Hp, Wp), where:
        - Hp is the patch height (rows),
        - Wp is the patch width (columns).

    stride : tuple of int
        The stride between patches (stride_vertical, stride_horizontal), where:
        - stride_vertical determines how many pixels to move down after each row,
        - stride_horizontal determines how many pixels to move right at each step.

    Returns
    -------
    torch.Tensor
        A tensor containing all extracted patches, of shape (M, C, Hp, Wp), where:
        - M is the total number of extracted patches (N * num_patches_per_image).

    Notes
    -----
    - This function works for multi-channel images.
    - Patches are extracted independently for each image in the batch.
    - No padding is applied; only fully-contained patches are extracted.
    """
    N, C, H, W = images.shape
    Hp, Wp = patch_size
    sV, sH = stride

    patches = []

    for i in range(0, H - Hp + 1, sV):  # vertical sliding
        for j in range(0, W - Wp + 1, sH):  # horizontal sliding
            patch = images[:, :, i:i+Hp, j:j+Wp]  # [N, C, Hp, Wp]
            patches.append(patch)

    patches = torch.cat(patches, dim=0)  # [M, C, Hp, Wp]

    return patches


def serialize_losses(losses, losses_weights):
    """
    Convert a list of loss functions and their corresponding weights into
    a serializable dictionary format.

    Args:
        losses (list): A list of loss function objects.
        losses_weights (list): A list of floats representing the weights for each loss.

    Returns:
        list: A list of dictionaries, each containing the name, weight, and
              any relevant parameters of the corresponding loss function.
    """
    losses_serializable = []

    for loss, weight in zip(losses, losses_weights):
        loss_entry = {
            "name": loss.__class__.__name__,
            "weight": weight,
            "params": {}
        }

        if isinstance(loss, GradientVariance):
            loss_entry["params"] = {
                "patch_size": loss.patch_size
            }
        
        if isinstance(loss, HistogramLoss):
            loss_entry["params"] = {
                "num_bins": loss.num_bins
            }

        losses_serializable.append(loss_entry)

    return losses_serializable


def deserialize_losses(config_data, device=None):
    """
    Reconstruct a list of loss functions and their weights from a serialized config.

    Args:
        config_data (dict): Dictionary containing serialized loss config under the "losses" key.
                            Each entry must have "name", "weight", and optionally "params".
        device (torch.device or str, optional): Device to pass to custom loss functions if needed.

    Returns:
        tuple: (losses, loss_weights)
            - losses: list of initialized loss function objects.
            - loss_weights: list of floats corresponding to the weight of each loss.
    """
    losses = []
    loss_weights = []

    for loss_entry in config_data["losses"]:
        name = loss_entry["name"]
        weight = loss_entry["weight"]
        params = loss_entry.get("params", {})

        if name == "MSELoss":
            losses.append(torch.nn.MSELoss())
        elif name == "GradientVariance":
            # Ensure device is passed if needed
            losses.append(GradientVariance(**params, device=device))
        elif name == 'HistogramLoss':
            losses.append(HistogramLoss(**params))
        else:
            raise ValueError(f"Unsupported loss function: {name}")

        loss_weights.append(weight)

    return losses, loss_weights
