import os
from pathlib import Path
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as patches # type: ignore
import numpy as np
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split # type: ignore
import torchvision.transforms.functional as functional_transforms # type: ignore
import csv
from tqdm import tqdm # type: ignore
import time

def tensor_to_image(tensor):
    # tensor: [channels, height, width] o [1, channels, height, width]
    img = tensor.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    img = (img - img.min()) / (img.max() - img.min())
    return img