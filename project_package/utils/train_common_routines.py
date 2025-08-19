# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import csv
import time
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Scientific and Visualization Libraries
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import matplotlib.patches as patches  # type: ignore
from skimage.metrics import structural_similarity as ssim
import lpips


# ───────────────────────────────────────────────────────────────────────────────
# 🌍 PyTorch and Torchvision
# ───────────────────────────────────────────────────────────────────────────────
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim  # type: ignore
from torch.utils.data import DataLoader, Dataset, random_split  # type: ignore
import torchvision.transforms.functional as functional_transforms  # type: ignore

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Utilities
# ───────────────────────────────────────────────────────────────────────────────
from tqdm import tqdm  # type: ignore



def multi_GPU_training(model):
    ''' If multiple GPU are available they are used for training, loading the model using nn.DataParallel. 
    '''
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        print("Model training to be done on multiple GPUs!")
    else:
        print("Model training to be done in only one GPU!")    

    return model    


def psnr(img1, img2, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    First we need to convert torch tensors to NumPy operable.
    It calculates psnr across last 3 tensor dimensions. For example, if we have a batch of tensors, it will the mean psnr along for 
    each tensor in the batch
    """
    img1 = img1.clamp(0,1).cpu().detach().numpy()
    img2 = img2.clamp(0,1).cpu().detach().numpy()
    img_diff = img1 - img2
    rmse = np.sqrt(np.mean((img_diff) ** 2, axis=(-3,-2,-1)))
    PSNR = np.mean(20 * np.log10(max_val / rmse))
    return PSNR                  

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim


def compute_ssim(img1, img2, max_val=1.):
    """
    Compute Structural Similarity Index Measure (the higher the better).
    SSIM is computed across last 3 dimensions, averaging over batch.
    Uses skimage's implementation for each image pair.
    First we need to convert torch tensors to NumPy operable.
    """
    img1 = img1.clamp(0, 1).cpu().detach().numpy()
    img2 = img2.clamp(0, 1).cpu().detach().numpy()
    
    # Assumes shape is (N, C, H, W)
    if img1.ndim == 4:
        batch_ssim = []
        for i in range(img1.shape[0]):
            # Move channel dimension to last for skimage (H, W, C)
            ssim_val = ssim(
                img1[i].transpose(1, 2, 0),
                img2[i].transpose(1, 2, 0),
                data_range=max_val,
                channel_axis=-1
            )
            batch_ssim.append(ssim_val)
        return np.mean(batch_ssim)
    
    elif img1.ndim == 3:
        # For single image (C, H, W)
        return ssim(
            img1.transpose(1, 2, 0),
            img2.transpose(1, 2, 0),
            data_range=max_val,
            channel_axis=-1
        )
    
    else:
        raise ValueError("Expected input shape (N, C, H, W) or (C, H, W)")



def init_small(m: nn.Module) -> None:
    """
    Custom initialization for convolutional layers (Conv2d).

    - Uses Kaiming Normal initialization (He et al., 2015), adapted for ReLU.
    - Scales the weights by 0.1 to reduce residual magnitude and improve
      stability in very deep residual networks (EDSR/RCAN style).
    - Sets bias (if present) to zero.

    Parameters
    ----------
    m : nn.Module
        PyTorch module. Only affects instances of nn.Conv2d.

    Example
    -------
    >>> model = RCAN(cfg)
    >>> model.apply(init_small)
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        m.weight.data.mul_(0.1)  # shrink residual contribution
        if m.bias is not None:
            nn.init.zeros_(m.bias)


import torch, torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

def charbonnier(x, y, eps=1e-6):
    return torch.sqrt((x - y).pow(2) + eps).mean()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.params = [p for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s)



def inference(model, inference_data, device):
    ''' Function that evaluate the model in inference task. Assuming that test dataset is small, it processes such data sequentially
    with nominal batch-size equal to 1
    model: Model to be used
    inference_data: Inference data (typically from the ouput of data_split function)
    device: Device where inference will be done
    Returns:
    final_psnr_orignal= PSNR of the input image with respect ground-truth
    final_psnr_model= PSNR of the output model image with respect ground-truth
    '''
    model.eval()
    running_psnr_original = 0.0
    running_psnr_model = 0.0
    with torch.no_grad():
        for i in range(len(inference_data)):
            data=inference_data[i]
            low_res_image = data[0].to(device)
            low_res_image =low_res_image.unsqueeze(0)
            truth_image = data[1].to(device)
                
            outputs = model(low_res_image)

            batch_psnr_model = psnr(truth_image, outputs)
            running_psnr_model += batch_psnr_model
            batch_psnr_original = psnr(truth_image, low_res_image)
            running_psnr_original += batch_psnr_original
    final_psnr_original = running_psnr_original/len(inference_data)
    final_psnr_model = running_psnr_model/len(inference_data)
    return final_psnr_original, final_psnr_model



def save_checkpoint(epoch, model, optimizer, loss, filename="checkpoint.pth"):
    '''Save checkpoints during training
    '''
    # If using DataParallel, remove it before saving
    model_checkpoint = model.module if hasattr(model, "module") else model
    checkpoint = {
        "epoch": epoch,
        "model_state": model_checkpoint.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }
    if os.path.exists(filename):
        os.remove(filename)
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")



def load_checkpoint(filename, model, optimizer, device):
    '''Load checkpoint for resume training
    '''

    checkpoint = torch.load(filename, map_location=device)

    # Handle DataParallel if used
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    print(f"Resumed training from epoch {epoch}, Loss: {loss:.4f}")
    return epoch     


def load_trained_model(model,file_model_inference,device):
    ''' Load trained model
    model: Intialized model (identical in structure to the one trained)
    file_model_inference=String with the file where model trained is saved
    device: device to where the model should be loaded
    Return
    model: Loaded model
    '''
    state_dict = torch.load(file_model_inference, map_location=device)
    model.load_state_dict(state_dict)  
    
    return model

# Instanciá la red LPIPS solo una vez afuera si querés reutilizarla
_lpips_model = lpips.LPIPS(net='alex')  # O 'vgg', 'squeeze'

def compute_lpips(img1, img2, net='alex'):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    Accepts images in [0, 1] range and computes perceptual similarity.
    Assumes img1 and img2 are tensors of shape (N, 3, H, W).
    Returns mean LPIPS score across the batch.
    """
    assert img1.shape == img2.shape, "Input images must have the same shape"
    assert img1.shape[1] == 3, "LPIPS expects 3-channel RGB images"

    # Scale from [0,1] to [-1,1]
    img1 = img1 * 2 - 1
    img2 = img2 * 2 - 1

    # Ensure on same device
    model = _lpips_model.to(img1.device)
    
    with torch.no_grad():
        d = model(img1, img2)

    return d.mean().item()


def test_model_single_images(model,input_image, ground_truth_image,device, fig, rows=1,columns=1, number_plot=1):
    '''Function that test model on single images and return a plot image
    model: Model to be test
    input_image: Model input image
    ground_truth_image: Ground truth image
    device: Device where computation are to be made
    fig: Figure to which the plot will be added
    rows: Total number of rows of the figure to which plot will be added
    columns: Total number of columns of the figure to which plot will be added
    number_plot: Number of plot added
    
    Returns
    img_model: Model output
    psnr_value_orig: Original PSNR value
    psnr_value_model: PSNR value for model output
    fig: Figure with added plot
    '''


    model.eval()
    with torch.no_grad():
        img = input_image.to(device)
        img = img.unsqueeze(0)
        img_model = model(img)
        psnr_value_model = psnr(ground_truth_image, img_model)
        psnr_value_orig = psnr(input_image, ground_truth_image)

    fig.add_subplot(rows, columns, number_plot) 
    plt.imshow(img_model.cpu().squeeze(0).permute(1,2,0).clamp(0,1).detach().numpy()) 

    return img_model, psnr_value_orig, psnr_value_model, fig


def show_image_with_patch(img,size_patch,top_y, left_x,title_main_image):
    '''Plot original image and patch
    size_patch: Size of patch
    top_y: Patch top y coordinate
    left_x: Path left x coordinate
    title_main_image: String with title for main image
    '''

    rect1 = patches.Rectangle((left_x, top_y), size_patch, size_patch, linewidth=2, edgecolor='r', facecolor='none')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img.permute(1,2,0).clamp(0,1).detach().numpy())
    ax[0].add_patch(rect1)
    ax[0].set_title(title_main_image)
    ax[0].axis("off")
    patch_low = img.permute(1,2,0).clamp(0,1).detach().numpy()[top_y:top_y+size_patch,left_x:left_x+size_patch]
    # Add the patch to the Axes
    # Show the plot
    ax[1].imshow(patch_low)
    ax[1].set_title("Zoomed Patch")
    ax[1].axis("off")
    plt.show()