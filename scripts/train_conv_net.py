# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import sys
import time
import csv
import json
from pathlib import Path

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Scientific & Data Libraries
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports (PyTorch)
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.optim as optim

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils import train_common_routines as tcr
from project_package.conv_net.ConvNet_model import SRCNN
from project_package.dataset_manager.webdataset_dataset import PtWebDataset

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
model_selection = 'large'
epochs = 200
lr = 1e-5
batch_size = 32

# ───────────────────────────────────────────────────────────────────────────────
# 📁 Paths Setup
# ───────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))

dataset = 'messi' # Select Dataset
dataset_folder = os.path.join(project_dir, 'datasets', dataset)
tar_path = "file:" + dataset_folder.replace("\\", "/")
metadata_path = os.path.join(dataset_folder, 'metadata.json')

with open(metadata_path, "r") as f:
    metadata = json.load(f)

train_samples = metadata["splits"]["train"]["num_samples"]
val_samples = metadata["splits"]["val"]["num_samples"]
test_samples = metadata["splits"]["test"]["num_samples"]

# Results folder and files
results_folder = os.path.join(project_dir, 'results', 'Conv_Net')
os.makedirs(results_folder, exist_ok=True)

loss_png_file = os.path.join(results_folder, f"loss_lr={lr}_batch_size={batch_size}_model={model_selection}.png")
psnr_png_file = os.path.join(results_folder, f"psnr_lr={lr}_batch_size={batch_size}_model={model_selection}.png")
final_model_pth_file = os.path.join(results_folder, f"model_lr={lr}_batch_size={batch_size}_model={model_selection}.pth")
file_training_losses = os.path.join(results_folder, f"training_losses_lr={lr}_batch_size={batch_size}_model={model_selection}.csv")

# ───────────────────────────────────────────────────────────────────────────────
# 🚀 Training Pipeline
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 🧠 Model Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = SRCNN(model_selection).to(device)
    print("The model:")
    print(model)

    model.count_parameters()
    print(f"Total Parameters: {model.total_params:,}")
    print(f"Trainable Parameters: {model.trainable_params:,}")

    model = tcr.multi_GPU_training(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 📊 Dataset and DataLoaders
    dataset_train = PtWebDataset(tar_path + '/train.tar', length=train_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
    dataset_val = PtWebDataset(tar_path + '/val.tar', length=val_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
    dataset_test = PtWebDataset(tar_path + '/test.tar', length=test_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)

    dataloader_train = dataset_train.get_dataloader(num_workers=0)
    dataloader_val = dataset_val.get_dataloader(num_workers=0)
    dataloader_test = dataset_test.get_dataloader(num_workers=0)

    # 🏋️ Training Loop
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []

    start = time.time()
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} of {epochs}")

        train_epoch_loss, train_epoch_psnr = tcr.train(model, dataloader_train, optimizer, tcr.compute_loss_MSE, device, train_samples)
        val_epoch_loss, val_epoch_psnr = tcr.validate(model, dataloader_val, epoch, tcr.compute_loss_MSE, device, val_samples)

        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")

        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)

        if epoch % 5 == 0:
            checkpoint_path = os.path.join(results_folder, f"checkpoint_epoch_{epoch}_lr={lr}_batch_size={batch_size}_model={model_selection}.pth")
            tcr.save_checkpoint(epoch, model, optimizer, train_loss, filename=checkpoint_path)

        with open(file_training_losses, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([train_epoch_loss, train_epoch_psnr, val_epoch_loss, val_epoch_psnr])

    end = time.time()
    print(f"\n✅ Finished training in: {(end - start) / 60:.2f} minutes")

    # 📈 Loss Plot
    plt.figure(figsize=(10, 7))
    plt.plot(10 * np.log10(train_loss), color='orange', label='Train Loss')
    plt.plot(10 * np.log10(val_loss), color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (dB)')
    plt.legend()
    if os.path.exists(loss_png_file):
        os.remove(loss_png_file)
    plt.savefig(loss_png_file)
    plt.show()

    # 📈 PSNR Plot
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='Train PSNR (dB)')
    plt.plot(val_psnr, color='blue', label='Validation PSNR (dB)')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    if os.path.exists(psnr_png_file):
        os.remove(psnr_png_file)
    plt.savefig(psnr_png_file)
    plt.show()

    # 💾 Save Final Model
    print('\n💾 Saving model...')
    if os.path.exists(final_model_pth_file):
        os.remove(final_model_pth_file)
    model = model.module if hasattr(model, "module") else model
    torch.save(model.state_dict(), final_model_pth_file)
