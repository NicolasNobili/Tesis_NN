from torch import nn
import torch
import torch.nn.functional as F

def mse(output, target):
    criterion = nn.MSELoss()
    return criterion(output, target)