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
# from sen2venus_routines import construct_full_data, read_file_list_train_data, load_files_tensor_data, extract_patches, load_files_tensor_data_patched
# from train_common_routines import multi_GPU_training, psnr, train, validate, save_checkpoint, compute_loss_MSE, data_split


class Tensor_images_dataset(Dataset):
    '''Class for management of data for training. Load from corresponding files generated with load_files_tensor_data
    '''
    def __init__(self, file_path_low_res,file_path_high_res):
        self.file_path_low_res = file_path_low_res
        self.file_path_high_res = file_path_high_res
        self.data_low_res = torch.load(file_path_low_res, weights_only=False, map_location="cpu")  
        self.data_high_res = torch.load(file_path_high_res, weights_only=False, map_location="cpu")  
          
    def __len__(self):
        return len(self.data_low_res)
        
    def __getitem__(self, idx):
        image_low_res= self.data_low_res[idx]
        image_truth_high_res = self.data_high_res[idx]
        return image_low_res, image_truth_high_res
            


class SRCNN_small(nn.Module):
    ''' Class for the NN model. 
    model_selection=String that is used for model selection. If equal to "small" we will use a small model. If "large" it will
                    be a large one
    '''
    def __init__(self,model_selection='small'):
        super(SRCNN_small, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2, padding_mode='replicate')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def count_parameters(self):
        self.total_params = sum(p.numel() for p in self.parameters())  # All parameters
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Only trainable ones
        return self.total_params, self.trainable_params     
    

class SRCNN_large(nn.Module):
    ''' Class for the NN model. 
    model_selection=String that is used for model selection. If equal to "small" we will use a small model. If "large" it will
                    be a large one
    '''
    def __init__(self,model_selection='small'):
        super(SRCNN_large, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4, padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=1, padding=2, padding_mode='replicate')
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(16, 3, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):  
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x

    def count_parameters(self):
        self.total_params = sum(p.numel() for p in self.parameters())  # All parameters
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Only trainable ones
        return self.total_params, self.trainable_params     