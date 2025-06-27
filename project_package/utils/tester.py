# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import sys
import os
import time
import csv
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
import logging


# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports (PyTorch)
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
# Add custom project folder to system path to enable local module imports
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils.train_common_routines import psnr

class Tester:
    """
    A class for evaluating super-resolution models on a test dataset.

    This class computes average loss and PSNR on the test set and optionally visualizes
    a few sample results, displaying input (low-res), model output (super-res), and target (high-res) images.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to be evaluated.
    device : torch.device
        The device (CPU or GPU) to run the evaluation on.
    compute_loss : callable
        The loss function used for evaluation.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    test_samples : int
        Total number of samples in the test set.
    checkpoint_path : str, optional
        Path to the model checkpoint to load weights from. Default is None.
    results_folder : str, optional
        Directory where visualized results will be saved. Default is None.
    visualize_count : int, optional
        Number of sample images to visualize. Default is 5.
    """

    def __init__(
        self,
        model,
        device,
        compute_loss,
        test_loader,
        test_samples,
        checkpoint_path=None,
        results_folder=None,
        visualize_count=5
    ):
        self.model = model.to(device)
        self.device = device
        self.compute_loss = compute_loss
        self.test_loader = test_loader
        self.test_samples = test_samples
        self.checkpoint_path = checkpoint_path
        self.results_folder = results_folder
        self.visualize_count = visualize_count

        if checkpoint_path:
            self.load_model()

        self.model.eval()

    def load_model(self):
        """
        Loads the model weights from a checkpoint file.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model_state = checkpoint["model_state"]

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(model_state)
        else:
            self.model.load_state_dict(model_state)

        print(f"[INFO] Loaded model from: {self.checkpoint_path}")

    def evaluate(self):
        """
        Evaluates the model on the test set.

        Computes the average loss and PSNR over all test samples.

        Returns
        -------
        avg_loss : float
            Average test loss.
        avg_psnr : float
            Average test PSNR in decibels (dB).
        """
        total_loss = 0.0
        total_psnr = 0.0
        total_psnr_lr = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Ensure that inputs have the same size as outputs before computing PSNR
                if inputs.shape[-2:] != outputs.shape[-2:]:
                    inputs_resized = F.interpolate(inputs, size=outputs.shape[-2:], mode='bicubic', align_corners=False)
                else:
                    inputs_resized = inputs

                if targets.shape[-2:] != outputs.shape[-2:]:
                    targets_resized = F.interpolate(targets, size=outputs.shape[-2:], mode='bicubic', align_corners=False)
                else:
                    targets_resized = targets

                #loss = self.compute_loss(outputs, targets_resized)
                loss =0
                for j in range(len(self.compute_loss)):
                    loss += self.compute_loss[j](outputs, targets)
                batch_size = inputs.size(0)

                total_loss += loss.item() * batch_size
                total_psnr_lr += psnr(targets_resized, inputs_resized) * batch_size
                total_psnr += psnr(targets_resized, outputs) * batch_size

                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_psnr = total_psnr / total_samples
        avg_psnr_lr = total_psnr_lr/ total_samples

        print(f"\n [RESULT] Test Loss: {avg_loss:.4f}")
        print(f"[RESULT] Test PSNR: {avg_psnr:.2f} dB")
        print(f'[RESULT] Bicubic: {avg_psnr_lr:.2f} db')

        return avg_loss, avg_psnr

    def visualize_results(self):
        """
        Visualizes a few predictions made by the model.

        For each selected sample, it displays:
        - Input (low-resolution)
        - Model output (super-resolution)
        - Ground truth (high-resolution)

        The images are saved to disk and also shown via matplotlib.
        """
        print(f"\n [INFO] Visualizing {self.visualize_count} test samples...")
        os.makedirs(self.results_folder, exist_ok=True)
        shown = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                for i in range(inputs.size(0)):
                    if shown >= self.visualize_count:
                        return

                    input_img = inputs[i].unsqueeze(0)   # Shape: (1, C, H, W)
                    output_img = outputs[i].unsqueeze(0)
                    target_img = targets[i]              # No resize for target

                    # Resize input to match output size if needed
                    if input_img.shape[-2:] != output_img.shape[-2:]:
                        input_img = F.interpolate(input_img, size=output_img.shape[-2:], mode='bicubic', align_corners=False)

                    # Convert tensors to PIL images
                    input_img = to_pil_image(input_img.squeeze(0).cpu().clamp(0, 1))
                    output_img = to_pil_image(output_img.squeeze(0).cpu().clamp(0, 1))
                    target_img = to_pil_image(target_img.cpu().clamp(0, 1))

                    # Plot and save the input, output, and target images
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(input_img)
                    axs[0].set_title("Input (Low-Res)")
                    axs[1].imshow(output_img)
                    axs[1].set_title("Output (Super-Res)")
                    axs[2].imshow(target_img)
                    axs[2].set_title("Target (High-Res)")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()

                    # Save the visualization to file
                    output_path = os.path.join(self.results_folder, f"sample_{shown + 1}.png")
                    plt.savefig(output_path)
                    plt.show()
                    shown += 1
