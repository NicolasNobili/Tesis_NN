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

class Tester_MS_aux:
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

        Computes the mean and standard deviation (sample estimator, ddof=1)
        of the loss, PSNR, SSIM, and LPIPS over all test samples.

        Returns
        -------
        avg_loss : tuple(float, float)
            Mean and std of total test loss.
        avg_loss_vec : list[tuple(float, float)]
            Mean and std per loss component.
        avg_psnr : tuple(float, float)
            Mean and std of test PSNR (dB).
        avg_psnr_lr : tuple(float, float)
            Mean and std of PSNR of low-res inputs (bicubic).
        avg_ssim : tuple(float, float)
            Mean and std of SSIM.
        avg_lpips : tuple(float, float)
            Mean and std of LPIPS.
        """
        # Guardar valores individuales para desviación estándar
        losses, psnrs, psnrs_lr, ssims, lpips_vals = [], [], [], [], []
        loss_vecs = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Convertir BGR → RGB
                inputs_rgb = inputs[:, [2, 1, 0], :, :]

                # Forward
                outputs = self.model(inputs_rgb)

                # Ajustar tamaño si es necesario
                if inputs_rgb.shape[-2:] != outputs.shape[-2:]:
                    inputs_resized = F.interpolate(
                        inputs_rgb, size=outputs.shape[-2:], mode='bicubic', align_corners=False
                    )
                else:
                    inputs_resized = inputs_rgb

                # Calcular pérdidas
                loss = 0
                loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)
                for j in range(len(self.compute_loss)):
                    loss_j = self.loss_weights[j] * self.compute_loss[j](outputs, targets)
                    loss += loss_j
                    loss_vec[j] = loss_j.item()

                # Guardar valores de batch
                losses.append(loss.item())
                loss_vecs.append(loss_vec)

                # Métricas
                psnrs_lr.append(psnr(targets, inputs_resized))
                psnrs.append(psnr(targets, outputs))
                ssims.append(compute_ssim(targets, outputs))
                lpips_vals.append(compute_lpips(targets, outputs))

        # Convertir listas a arrays
        losses = np.array(losses)
        psnrs = np.array(psnrs)
        psnrs_lr = np.array(psnrs_lr)
        ssims = np.array(ssims)
        lpips_vals = np.array(lpips_vals)
        loss_vecs = np.array(loss_vecs)

        # Calcular medias y desviaciones (usando estimador muestral)
        avg_loss = (losses.mean(), losses.std(ddof=1))
        avg_loss_vec = [(loss_vecs[:, i].mean(), loss_vecs[:, i].std(ddof=1)) for i in range(loss_vecs.shape[1])]
        avg_psnr = (psnrs.mean(), psnrs.std(ddof=1))
        avg_psnr_lr = (psnrs_lr.mean(), psnrs_lr.std(ddof=1))
        avg_ssim = (ssims.mean(), ssims.std(ddof=1))
        avg_lpips = (lpips_vals.mean(), lpips_vals.std(ddof=1))

        # Mostrar resultados
        print(f"\n[RESULT] Test Loss: {avg_loss[0]:.4f} ± {avg_loss[1]:.4f}")
        print(f"[RESULT] Test PSNR: {avg_psnr[0]:.2f} ± {avg_psnr[1]:.2f} dB")
        print(f"[RESULT] Bicubic PSNR: {avg_psnr_lr[0]:.2f} ± {avg_psnr_lr[1]:.2f} dB")
        print(f"[RESULT] SSIM: {avg_ssim[0]:.4f} ± {avg_ssim[1]:.4f}")
        print(f"[RESULT] LPIPS: {avg_lpips[0]:.4f} ± {avg_lpips[1]:.4f}")

        return avg_loss, avg_loss_vec, avg_psnr, avg_psnr_lr, avg_ssim, avg_lpips


    def visualize_results(self, folder_path=None):
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

        # Carpeta de salida
        test_images_root = folder_path or os.path.join(self.results_folder, 'test_images')
        os.makedirs(test_images_root, exist_ok=True)

        shown = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Convertir a RGB antes del forward
                inputs_rgb = inputs[:, [2, 1, 0], :, :]
                outputs = self.model(inputs_rgb)

                for batch_index in range(inputs.size(0)):
                    if shown >= self.visualize_count:
                        return

                    # Subcarpeta por muestra
                    sample_folder = os.path.join(test_images_root, f"sample_{shown + 1}")
                    os.makedirs(sample_folder, exist_ok=True)

                    # Extraer tensores individuales
                    tensor_low_4 = inputs[batch_index]
                    tensor_low = tensor_low_4[[2, 1, 0], :, :]
                    tensor_out = outputs[batch_index]
                    tensor_high = targets[batch_index]

                    input_img = tensor_low.unsqueeze(0)
                    output_img = tensor_out.unsqueeze(0)
                    target_img = tensor_high

                    # Asegurar tamaño compatible
                    if input_img.shape[-2:] != output_img.shape[-2:]:
                        input_img = F.interpolate(input_img, size=output_img.shape[-2:], mode='bicubic', align_corners=False)

                    # Convertir a imágenes PIL
                    input_pil = to_pil_image(input_img.squeeze(0).cpu().clamp(0, 1))
                    output_pil = to_pil_image(output_img.squeeze(0).cpu().clamp(0, 1))
                    target_pil = to_pil_image(target_img.cpu().clamp(0, 1))

                    # Calcular PSNR
                    psnr_lr = psnr(target_img.unsqueeze(0).to(self.device), input_img.to(self.device)).item()
                    psnr_sr = psnr(target_img.unsqueeze(0).to(self.device), output_img.to(self.device)).item()

                    # ───────────────────────────────
                    # FIGURA PRINCIPAL (LR, SR, HR)
                    # ───────────────────────────────
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    imgs = [input_pil, output_pil, target_pil]
                    titles = ["Input (Low-Res)", "Output (Super-Res)", "Target (High-Res)"]
                    psnrs = [psnr_lr, psnr_sr, None]

                    for ax, img, title, val in zip(axs, imgs, titles, psnrs):
                        ax.imshow(img)
                        ax.set_title(title, fontsize=11)
                        ax.axis("off")
                        if val is not None:
                            # PSNR centrado justo debajo
                            ax.text(
                                0.5, -0.06, f"PSNR: {val:.2f} dB",
                                ha='center', va='top', transform=ax.transAxes, fontsize=10
                            )

                    plt.subplots_adjust(wspace=0.05, hspace=0.2)
                    plt.savefig(os.path.join(sample_folder, "comparison_full.png"),
                                bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # ───────────────────────────────
                    # GUARDAR IMÁGENES INDIVIDUALES
                    # ───────────────────────────────
                    sample_folder_originals = os.path.join(sample_folder, "originals")
                    os.makedirs(sample_folder_originals, exist_ok=True)
                    input_pil.save(os.path.join(sample_folder_originals, "input_LR.png"))
                    output_pil.save(os.path.join(sample_folder_originals, "output_SR.png"))
                    target_pil.save(os.path.join(sample_folder_originals, "target_HR.png"))

                    # ───────────────────────────────
                    # OPCIONAL: PARCHES
                    # ───────────────────────────────
                    if self.patching:
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

                            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
                            for ax, img, title in zip(axs, [pil_low, pil_out, pil_high],
                                                    ["Patch Low-Res", "Patch Super-Res", "Patch High-Res"]):
                                ax.imshow(img)
                                ax.set_title(title, fontsize=10)
                                ax.axis("off")

                            plt.tight_layout()
                            plt.savefig(os.path.join(sample_folder, f"patch_comparison_{j + 1}.png"),
                                        bbox_inches='tight', dpi=200)
                            plt.close(fig)

                    shown += 1


    def test_single_image(self, input_tensor, target_tensor):
        """
        Evalúa una única imagen de entrada (tensor) y devuelve:
        - Imagen superresuelta,
        - Imagen bicúbica,
        - Imagen de referencia (target),
        - PSNR, SSIM, LPIPS para bicúbica y SR.

        Parámetros:
        ----------
        input_tensor : torch.Tensor
            Imagen de baja resolución. Shape: [C, H, W]
        target_tensor : torch.Tensor
            Imagen de alta resolución (ground truth). Shape: [C, H, W]

        Retorna:
        -------
        dict con claves: sr_image, bicubic_image, target_image,
                        psnr_sr, ssim_sr, lpips_sr,
                        psnr_bicubic, ssim_bicubic, lpips_bicubic
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            target_tensor = target_tensor.unsqueeze(0).to(self.device)

            inputs_rgb = input_tensor[[2, 1, 0], :, :]
            output_tensor = self.model(inputs_rgb )

            # Bicubic upscale
            input_rgb = input_tensor[[2, 1, 0], :, :]
            bicubic_tensor = F.interpolate(input_rgb, size=output_tensor.shape[-2:], mode='bicubic', align_corners=False)

            # Métricas
            psnr_sr = psnr(target_tensor, output_tensor).item()
            ssim_sr = compute_ssim(target_tensor, output_tensor).item()
            lpips_sr = compute_lpips(target_tensor, output_tensor).item()

            psnr_bicubic = psnr(target_tensor, bicubic_tensor).item()
            ssim_bicubic = compute_ssim(target_tensor, bicubic_tensor).item()
            lpips_bicubic = compute_lpips(target_tensor, bicubic_tensor).item()

            return {
                'sr_image': output_tensor.squeeze(0).cpu(),
                'bicubic_image': bicubic_tensor.squeeze(0).cpu(),
                'target_image': target_tensor.squeeze(0).cpu(),
                'psnr_sr': psnr_sr,
                'ssim_sr': ssim_sr,
                'lpips_sr': lpips_sr,
                'psnr_bicubic': psnr_bicubic,
                'ssim_bicubic': ssim_bicubic,
                'lpips_bicubic': lpips_bicubic,
            }
