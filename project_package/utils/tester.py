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

from project_package.utils.train_common_routines import psnr, compute_lpips, compute_ssim
from project_package.utils.utils import extract_patches

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
        The loss functions used for evaluation.
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
        loss_weights,
        test_loader,
        test_samples,
        checkpoint_path=None,
        results_folder=None,
        visualize_count=5,
        patching=False,
        patch_size=None,
        stride=None,
    ):
        self.model = model.to(device)
        self.device = device
        self.compute_loss = compute_loss
        self.loss_weights = loss_weights
        self.test_loader = test_loader
        self.test_samples = test_samples
        self.checkpoint_path = checkpoint_path
        self.results_folder = results_folder
        self.visualize_count = visualize_count
        self.patching = patching
        self.patch_size = patch_size
        self.stride = stride

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

        Computes the average loss, PSNR, SSIM, and LPIPS over all test samples.

        Returns
        -------
        avg_loss : float
            Average test loss.
        avg_loss_vec : np.ndarray
            Average loss vector per loss component.
        avg_psnr : float
            Average test PSNR in decibels (dB).
        avg_psnr_lr : float
            Average PSNR of low-res inputs (bicubic).
        avg_ssim : float
            Average SSIM.
        avg_lpips : float
            Average LPIPS.
        """
        total_loss = 0.0
        total_psnr = 0.0
        total_psnr_lr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                if self.patching:
                    inputs = extract_patches(
                        images=inputs, 
                        patch_size=self.patch_size['low'], 
                        stride=self.stride['low']
                    )
                    outputs = extract_patches(
                        images=outputs, 
                        patch_size=self.patch_size['high'], 
                        stride=self.stride['high']
                    )
                    targets = extract_patches(
                        images=targets, 
                        patch_size=self.patch_size['high'], 
                        stride=self.stride['high']
                    )

                # Resize inputs and targets if needed to match outputs spatial size
                if inputs.shape[-2:] != outputs.shape[-2:]:
                    inputs_resized = F.interpolate(inputs, size=outputs.shape[-2:], mode='bicubic', align_corners=False)
                else:
                    inputs_resized = inputs

                if targets.shape[-2:] != outputs.shape[-2:]:
                    targets_resized = F.interpolate(targets, size=outputs.shape[-2:], mode='bicubic', align_corners=False)
                else:
                    targets_resized = targets

                # Compute losses
                loss = 0
                loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)
                for j in range(len(self.compute_loss)):
                    loss_j = self.loss_weights[j] * self.compute_loss[j](outputs, targets_resized)
                    loss += loss_j
                    loss_vec[j] = loss_j.item()

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                for j in range(len(self.compute_loss)):
                    total_loss_vec[j] += loss_vec[j] * batch_size

                # Metrics: PSNR, SSIM, LPIPS
                total_psnr_lr += psnr(targets_resized, inputs_resized) * batch_size
                total_psnr += psnr(targets_resized, outputs) * batch_size
                total_ssim += compute_ssim(targets_resized, outputs) * batch_size
                total_lpips += compute_lpips(targets_resized, outputs) * batch_size

                total_samples += batch_size 

        avg_loss = total_loss / total_samples
        avg_loss_vec = total_loss_vec / total_samples
        avg_psnr = total_psnr / total_samples
        avg_psnr_lr = total_psnr_lr / total_samples
        avg_ssim = total_ssim / total_samples
        avg_lpips = total_lpips / total_samples

        print(f"\n[RESULT] Test Loss: {avg_loss:.4f}")
        print(f"[RESULT] Test PSNR: {avg_psnr:.2f} dB")
        print(f"[RESULT] Bicubic PSNR: {avg_psnr_lr:.2f} dB")
        print(f"[RESULT] SSIM: {avg_ssim:.4f}")
        print(f"[RESULT] LPIPS: {avg_lpips:.4f}")

        return avg_loss, avg_loss_vec, avg_psnr, avg_psnr_lr, avg_ssim, avg_lpips


    def visualize_results(self):
        """
        Visualizes predictions with optional patch-level comparisons.

        For each test sample:
        - Creates a subfolder `test_images/sample_{i}/`
        - Saves a comparison plot of the full image (low-res, super-res, high-res)
        - If patching is enabled:
            - Extracts patches
            - Saves one comparison plot per patch (low-res, super-res, high-res)
        """
        print(f"\n[INFO] Visualizing {self.visualize_count} test samples...")
        test_images_root = os.path.join(self.results_folder,'test_images')
        os.makedirs(test_images_root, exist_ok=True)
        shown = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                for batch_index in range(inputs.size(0)):
                    if shown >= self.visualize_count:
                        return

                    # Subfolder for this sample
                    sample_folder = os.path.join(test_images_root, f"sample_{shown + 1}")
                    os.makedirs(sample_folder, exist_ok=True)

                    # Extract tensors
                    tensor_low = inputs[batch_index]
                    tensor_out = outputs[batch_index]
                    tensor_high = targets[batch_index]

                    # Ensure shapes match
                    input_img = tensor_low.unsqueeze(0)
                    output_img = tensor_out.unsqueeze(0)
                    target_img = tensor_high

                    if input_img.shape[-2:] != output_img.shape[-2:]:
                        input_img = F.interpolate(input_img, size=output_img.shape[-2:], mode='bicubic', align_corners=False)

                    # Convert to PIL
                    input_pil = to_pil_image(input_img.squeeze(0).cpu().clamp(0, 1))
                    output_pil = to_pil_image(output_img.squeeze(0).cpu().clamp(0, 1))
                    target_pil = to_pil_image(target_img.cpu().clamp(0, 1))

                    # Save full image comparison
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(input_pil)
                    axs[0].set_title("Input (Low-Res)")
                    axs[1].imshow(output_pil)
                    axs[1].set_title("Output (Super-Res)")
                    axs[2].imshow(target_pil)
                    axs[2].set_title("Target (High-Res)")
                    for ax in axs:
                        ax.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(sample_folder, "comparison_full.png"))
                    plt.close(fig)

                    if self.patching:
                        # Optional patching
                        patches_low = extract_patches(
                            images=tensor_low.unsqueeze(0), 
                            patch_size=self.patch_size['low'], 
                            stride=self.stride['low']
                        )
                        patches_out = extract_patches(
                            images=tensor_out.unsqueeze(0), 
                            patch_size=self.patch_size['high'], 
                            stride=self.stride['high']
                        )
                        patches_high = extract_patches(
                            images=tensor_high.unsqueeze(0), 
                            patch_size=self.patch_size['high'], 
                            stride=self.stride['high']
                        )

                        num_patches = patches_low.size(0)

                        for j in range(num_patches):
                            input_img = patches_low[j].unsqueeze(0)
                            output_img = patches_out[j].unsqueeze(0)
                            target_img = patches_high[j]

                            if input_img.shape[-2:] != output_img.shape[-2:]:
                                input_img = F.interpolate(input_img, size=output_img.shape[-2:], mode='bicubic', align_corners=False)

                            pil_low = to_pil_image(input_img.squeeze(0).cpu().clamp(0, 1))
                            pil_out = to_pil_image(output_img.squeeze(0).cpu().clamp(0, 1))
                            pil_high = to_pil_image(target_img.cpu().clamp(0, 1))

                            # Save patch comparison
                            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                            axs[0].imshow(pil_low)
                            axs[0].set_title("Patch Low-Res")
                            axs[1].imshow(pil_out)
                            axs[1].set_title("Patch Super-Res")
                            axs[2].imshow(pil_high)
                            axs[2].set_title("Patch High-Res")
                            for ax in axs:
                                ax.axis('off')
                            plt.tight_layout()
                            plt.savefig(os.path.join(sample_folder, f"patch_comparison_{j + 1}.png"))
                            plt.close(fig)

                    shown += 1


                    
