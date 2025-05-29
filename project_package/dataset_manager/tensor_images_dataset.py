import os
from pathlib import Path
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split # type: ignore
import torchvision.transforms.functional as functional_transforms # type: ignore
import csv
from tqdm import tqdm # type: ignore
import time


class Tensor_images_dataset(Dataset):
    '''Class for management of data for training. Load from corresponding files generated with load_files_tensor_data
    '''
    def __init__(self, file_path_low_res,file_path_high_res):
        self.file_path_low_res = file_path_low_res
        self.file_path_high_res = file_path_high_res
        self.data_low_res = torch.load(file_path_low_res, weights_only=False, map_location="cpu").float()  
        self.data_high_res = torch.load(file_path_high_res, weights_only=False, map_location="cpu").float()
          
    def __len__(self):
        return len(self.data_low_res)
        
    def __getitem__(self, idx):
        print(self.data_low_res[idx].dtype) 
        resized_image_low_res=F.interpolate(self.data_low_res[idx].unsqueeze(0), size=(256, 256), mode='bicubic', align_corners=False).squeeze(0)
        image_truth_high_res = self.data_high_res[idx]
        return resized_image_low_res, image_truth_high_res
            
