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
# ⬇️ Downsampling Block (MaxPooling)
# ───────────────────────────────────────────────────────────────────────────────

class down_block(nn.Module):
    """
    A basic downsampling block using MaxPooling.
    """
    def __init__(self, scale):
        super(down_block, self).__init__()
        self.mp= nn.MaxPool2d(scale)

    def forward(self, x):
        x = self.mp(x)
        return x
    
    
# ───────────────────────────────────────────────────────────────────────────────
# ⬆️ Upsampling Block with optional bilinear or PixelShuffle
# ───────────────────────────────────────────────────────────────────────────────
    
class up_block(nn.Module):
    """
    Upsamples the feature map, applies a convolution, then concatenates with the corresponding skip connection.
    """
    def __init__(self, scale, in_channels, out_channels, bilinear=False, bias=True):
        super(up_block,self).__init__()
        m = []
        if bilinear: 
            self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else: 
            if (scale & (scale - 1)) == 0:  # If scale is power of 2
                for _ in range(int(math.log(scale, 2))):
                    m.append(default_conv(in_channels, 4 * in_channels, 3,bias=bias))
                    m.append(nn.PixelShuffle(2))
            else:
                raise NotImplementedError("Only power-of-two scales supported for PixelShuffle.")
            self.upscale = nn.Sequential(*m)

        self.conv = default_conv(in_channels=in_channels,out_channels=out_channels, kernel_size=3)

    def forward(self,x,z):
        x = self.upscale(x)
        x = self.conv(x)
        x = torch.cat([x, z], dim=1)
        return x
    

# ───────────────────────────────────────────────────────────────────────────────
# 🧱 Default Convolutional Block
# ───────────────────────────────────────────────────────────────────────────────

class default_block(nn.Module):
    """
    A block with 3 convolution layers and ReLU activations.
    """
    def __init__(self,in_channels,out_channels,kernel_size):
        super(default_block,self).__init__()
        self.layer = nn.Sequential(
            default_conv(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size),
            nn.ReLU(inplace=False),
            default_conv(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size),
            nn.ReLU(inplace=False),
            #default_conv(in_channels=out_channels,out_channels=out_channels,kernel_size=kernel_size),
            #nn.ReLU(inplace=False)
        )
    def forward(self,x):
        return self.layer(x)


# ───────────────────────────────────────────────────────────────────────────────
# *️⃣ Basic Upsampler Module
# ───────────────────────────────────────────────────────────────────────────────   

class upsampler(nn.Module):
    """
    Module to increase spatial resolution using bilinear or PixelShuffle upsampling.
    """
    def __init__(self, scale, n_feats,bilinear = False, bias=True):
        super(upsampler,self).__init__()
        m = []
        if bilinear: 
            self.upscale = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else: 
            if (scale & (scale - 1)) == 0:  # If scale is power of 2
                for _ in range(int(math.log(scale, 2))):
                    m.append(default_conv(n_feats, 4 * n_feats, 3,bias=bias))
                    m.append(nn.PixelShuffle(2))
            else:
                raise NotImplementedError
            self.upscale = nn.Sequential(*m)

    def forward(self, x, z=None):
        x = self.upscale(x)
        return x
    


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 UNet1: Upsampling After U-Net (Post-Upsampling Strategy)
# ───────────────────────────────────────────────────────────────────────────────

class UNet1(nn.Module):
    """
    U-Net architecture for super-resolution where the final output is upsampled after decoding.
    """
    def __init__(self, args):
        super(UNet1, self).__init__()
        self.n_channels = args.n_channels

        self.downconv_path = nn.ModuleList()
        self.downsample_path = nn.ModuleList()


        for i in range(len(args.n_channels)-1):
            in_ch = 3 if i == 0 else args.n_channels[i - 1]
            out_ch = args.n_channels[i]
            self.downconv_path.append(default_block(in_channels= in_ch, out_channels= out_ch, kernel_size=3))
            self.downsample_path.append(down_block(2))

        self.mid_conv = default_block(in_channels=args.n_channels[-2],out_channels=args.n_channels[-1], kernel_size=3)

        self.upsample_path = nn.ModuleList()
        self.upconv_path = nn.ModuleList()
        for i in range(len(args.n_channels)-1,0,-1):
            in_ch = args.n_channels[i]
            out_ch = args.n_channels[i-1]
            self.upsample_path.append(up_block(scale=2,in_channels=in_ch,out_channels=out_ch))
            self.upconv_path.append(default_block(in_channels=2*out_ch,out_channels=out_ch, kernel_size=3))

        m_tail = [
            upsampler(args.scale, args.n_channels[0]),
            default_conv(args.n_channels[0], args.n_colors, 3)
        ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x_list = []
        # Encoder
        for i in range(len(self.downsample_path)):
            x = self.downconv_path[i](x)
            x_list.append(x)
            x = self.downsample_path[i](x)

        # Bottleneck
        x = self.mid_conv(x)

        # Decoder
        for i in range(len(self.upsample_path)):
            skip = x_list[-(i + 1)] 
            x = self.upsample_path[i](x, skip)
            x = self.upconv_path[i](x)

        x = self.tail(x)
        return x
    
    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.total_params, self.trainable_params

    
# ───────────────────────────────────────────────────────────────────────────────
# 🧠 UNet2: Upsampling Before U-Net (Pre-Upsampling Strategy)
# ───────────────────────────────────────────────────────────────────────────────

class UNet2(nn.Module):
    """
    U-Net for super-resolution where input is upsampled before entering the encoder.
    """
    def __init__(self, args):
        super(UNet2, self).__init__()
        self.n_channels = args.n_channels

        self.input_layer = upsampler(scale=args.scale,n_feats=3)

        self.downconv_path = nn.ModuleList()
        self.downsample_path = nn.ModuleList()
        for i in range(len(args.n_channels)-1):
            in_ch = 3 if i == 0 else args.n_channels[i - 1]
            out_ch = args.n_channels[i]
            self.downconv_path.append(default_block(in_channels= in_ch, out_channels= out_ch, kernel_size=3))
            self.downsample_path.append(down_block(2))

        self.mid_conv = default_block(in_channels=args.n_channels[-2],out_channels=args.n_channels[-1], kernel_size=3)

        self.upsample_path = nn.ModuleList()
        self.upconv_path = nn.ModuleList()
        for i in range(len(args.n_channels)-1,0,-1):
            in_ch = args.n_channels[i]
            out_ch = args.n_channels[i-1]
            self.upsample_path.append(up_block(scale=2,in_channels=in_ch,out_channels=out_ch))
            self.upconv_path.append(default_block(in_channels=2*out_ch,out_channels=out_ch, kernel_size=3))

        self.last_layer = default_conv(in_channels=args.n_channels[0],out_channels=3,kernel_size=3)

    def forward(self, x):
        x = self.input_layer(x)

        x_list = []

        # Encoder
        for i in range(len(self.downsample_path)):
            x = self.downconv_path[i](x)
            x_list.append(x)
            x = self.downsample_path[i](x)

        # Bottleneck
        x = self.mid_conv(x)

        # Decoder
        for i in range(len(self.upsample_path)):
            skip = x_list[-(i + 1)] 
            x = self.upsample_path[i](x, skip)
            x = self.upconv_path[i](x)

        x = self.last_layer(x)
        return x
    
    def count_parameters(self):
        """
        Returns the total and trainable parameter count.
        """
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.total_params, self.trainable_params
    
    
# ───────────────────────────────────────────────────────────────────────────────
# 🧠 UNet CONFIG 
# ───────────────────────────────────────────────────────────────────────────────

class UNetConfig:
    def __init__(self, n_channels, scale, n_colors, rgb_range):
        self.n_channels = n_channels      # List of channels per layer (e.g. [64, 128, 256])
        self.scale = scale                # Super-resolution scale factor
        self.n_colors = n_colors          # Number of output color channels (e.g. 3 for RGB)
        self.rgb_range = rgb_range        # RGB value range (e.g. 255)

    def __repr__(self):
        return (f"UNetConfig(n_channels={self.n_channels}, "
                f"scale={self.scale}, n_colors={self.n_colors}, "
                f"rgb_range={self.rgb_range})")


