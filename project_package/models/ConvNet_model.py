# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import time
import csv
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
# 🧠 SRCNN Small Model
# ───────────────────────────────────────────────────────────────────────────────
class SRCNN_small(nn.Module):
    """
    Small version of the Super-Resolution Convolutional Neural Network (SRCNN).
    Consists of 3 convolutional layers for basic super-resolution tasks.
    """
    def __init__(self):
        super(SRCNN_small, self).__init__()
        self.conv1 = default_conv(in_channels=3,out_channels=64,kernel_size=9,padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = default_conv(in_channels=64,out_channels=32,kernel_size=3,padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = default_conv(in_channels=32,out_channels=3,kernel_size=5,padding_mode='replicate')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.total_params, self.trainable_params


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 SRCNN Large Model
# ───────────────────────────────────────────────────────────────────────────────
class SRCNN_large(nn.Module):
    """
    Larger variant of the SRCNN model with deeper structure.
    May improve performance on more complex super-resolution tasks.
    """
    def __init__(self):
        super(SRCNN_large, self).__init__()
        self.conv1 = default_conv(in_channels=3,out_channels=64,kernel_size=9,padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = default_conv(in_channels=64,out_channels=32,kernel_size=3,padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = default_conv(in_channels=16,out_channels=64,kernel_size=1,padding_mode='replicate')
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = default_conv(in_channels=16,out_channels=3,kernel_size=3,padding_mode='replicate')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x

    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
