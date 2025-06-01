# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
from pathlib import Path
import sys
import csv
import time

# Numerical and Data Handling
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────

# Add custom project directory to system path for local imports
sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

# Import custom modules for training routines, model, datasets, and utilities
from project_package.utils import train_common_routines2 as tcr
from project_package.conv_net.ConvNet_model import SRCNN
from project_package.dataset_manager.tensor_images_dataset import Tensor_images_dataset
from project_package.data_processing import sen2venus_routines as s2v
from project_package.utils import utils as utils

# ───────────────────────────────────────────────────────────────────────────────
# 🖥 Device Setup
# ───────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ───────────────────────────────────────────────────────────────────────────────
# ⚙️ Configuration Parameters
# ───────────────────────────────────────────────────────────────────────────────
dataset = 'my_dataset2'

# Training parameters
patched_images = 'no'  # Use patched images or not ('yes'/'no')
model_selection = 'large'  # Model size selection
epochs = 200
learning_rate = 1e-5
batch_size = 32
patch_size = 64
stride = 48
scale_value = 10000  # Scale factor for images (not used explicitly here but reserved)

# Dataset split ratios
train_ratio = 0.95
val_ratio = 0.04
test_ratio = 1 - train_ratio - val_ratio

# ───────────────────────────────────────────────────────────────────────────────
# 🗂 Paths Setup
# ───────────────────────────────────────────────────────────────────────────────
data_folder = Path(f'C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/datasets/{dataset}')
results_folder = Path('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results/Conv_Net')
results_folder.mkdir(parents=True, exist_ok=True)

# Filepaths for saving outputs
loss_plot_file = results_folder / f"loss_lr={learning_rate}_batch_size={batch_size}_model={model_selection}_patched_images={patched_images}.png"
psnr_plot_file = results_folder / f"psnr_lr={learning_rate}_batch_size={batch_size}_model={model_selection}_patched_images={patched_images}.png"
final_model_file = results_folder / f"model_lr={learning_rate}_batch_size={batch_size}_model={model_selection}_patched_images={patched_images}.pth"
training_losses_csv = results_folder / f"training_losses_lr={learning_rate}_batch_size={batch_size}_model={model_selection}_patched_images={patched_images}.csv"

# ───────────────────────────────────────────────────────────────────────────────
# 🔨 Model Initialization
# ───────────────────────────────────────────────────────────────────────────────
model = SRCNN(model_selection).to(device)
print("Model architecture:\n", model)

# Display parameter counts
model.count_parameters()
print(f"Total Parameters: {model.total_params:,}")
print(f"Trainable Parameters: {model.trainable_params:,}")

# Enable multi-GPU training if available
model = tcr.multi_GPU_training(model)

# Optimizer setup
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ───────────────────────────────────────────────────────────────────────────────
# 📊 Dataset Loading and DataLoader Creation
# ───────────────────────────────────────────────────────────────────────────────
if patched_images == "no":
    # Paths to unpatched dataset files (.pt)
    file_path_low_res = data_folder / '10m.pt'
    file_path_high_res = data_folder / '05m.pt'

    # Initialize dataset
    dataset = Tensor_images_dataset(file_path_low_res, file_path_high_res)

    # Split dataset into train, validation, and test
    train_data, val_data, test_data = tcr.data_split(dataset, train_ratio, val_ratio, test_ratio)

    # Create DataLoaders with batching and shuffling where appropriate
    dataloader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Training with unpatched images")

elif patched_images == "yes":
    # Paths to patched dataset files (.pt)
    file_path_low_res_train = data_folder / f'train_data_low_res_patched_patch_size={patch_size}_stride={stride}.pt'
    file_path_high_res_train = data_folder / f'train_data_high_res_patched_patch_size={patch_size}_stride={stride}.pt'
    dataset_train = Tensor_images_dataset(file_path_low_res_train, file_path_high_res_train)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    file_path_low_res_val = data_folder / f'val_data_low_res_patched_patch_size={patch_size}_stride={stride}.pt'
    file_path_high_res_val = data_folder / f'val_data_high_res_patched_patch_size={patch_size}_stride={stride}.pt'
    dataset_val = Tensor_images_dataset(file_path_low_res_val, file_path_high_res_val)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("Training with patched images")

# ───────────────────────────────────────────────────────────────────────────────
# 🏃 Training Loop
# ───────────────────────────────────────────────────────────────────────────────
train_loss, val_loss = [], []
train_psnr, val_psnr = [], []

start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    # Training step
    train_epoch_loss, train_epoch_psnr = tcr.train(model, dataloader_train, optimizer, tcr.compute_loss_MSE, device)

    # Validation step
    val_epoch_loss, val_epoch_psnr = tcr.validate(model, dataloader_val, epoch, tcr.compute_loss_MSE, device)

    # Log epoch metrics
    print(f"Train PSNR: {train_epoch_psnr:.3f}")
    print(f"Val PSNR: {val_epoch_psnr:.3f}")

    train_loss.append(train_epoch_loss)
    train_psnr.append(train_epoch_psnr)
    val_loss.append(val_epoch_loss)
    val_psnr.append(val_epoch_psnr)

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        checkpoint_path = results_folder / f"checkpoint_epoch_{epoch}_lr={learning_rate}_batch_size={batch_size}_model={model_selection}_patched_images={patched_images}.pth"
        tcr.save_checkpoint(epoch, model, optimizer, train_loss, filename=checkpoint_path)

    # Append training losses and PSNRs to CSV file
    with open(training_losses_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([train_epoch_loss, train_epoch_psnr, val_epoch_loss, val_epoch_psnr])

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"Finished training in: {elapsed_minutes:.3f} minutes")

# ───────────────────────────────────────────────────────────────────────────────
# 📈 Plot Training Results
# ───────────────────────────────────────────────────────────────────────────────

# Plot training and validation loss (converted to dB scale)
plt.figure(figsize=(10, 7))
plt.plot(10 * np.log10(train_loss), color='orange', label='Train Loss (dB)')
plt.plot(10 * np.log10(val_loss), color='red', label='Validation Loss (dB)')
plt.xlabel('Epochs')
plt.ylabel('Loss (dB)')
plt.legend()
if loss_plot_file.exists():
    loss_plot_file.unlink()
plt.savefig(loss_plot_file)
plt.show()

# Plot training and validation PSNR
plt.figure(figsize=(10, 7))
plt.plot(train_psnr, color='green', label='Train PSNR (dB)')
plt.plot(val_psnr, color='blue', label='Validation PSNR (dB)')
plt.xlabel('Epochs')
plt.ylabel('PSNR (dB)')
plt.legend()
if psnr_plot_file.exists():
    psnr_plot_file.unlink()
plt.savefig(psnr_plot_file)
plt.show()

# ───────────────────────────────────────────────────────────────────────────────
# 💾 Save Final Model
# ───────────────────────────────────────────────────────────────────────────────
print('Saving final model...')
if final_model_file.exists():
    final_model_file.unlink()

# If model was wrapped in DataParallel, unwrap before saving
model_to_save = model.module if hasattr(model, "module") else model
torch.save(model_to_save.state_dict(), final_model_file)
