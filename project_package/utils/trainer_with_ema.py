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
# 📚 Scientific & Data Libraries
# ───────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports (PyTorch)
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image


# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
# Add custom project folder to system path to enable local module imports
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils.train_common_routines import psnr


class Trainer_EMA:
    """
    A class to manage the training, validation, logging, and saving of a deep learning model.

    This class encapsulates the full training pipeline including epoch-level training/validation,
    checkpointing, plotting metrics, and saving the final trained model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to be trained.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    compute_loss : callable
        Loss functions used during training and validation.
    device : torch.device
        The device on which computations are performed (e.g., 'cpu' or 'cuda').

    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.

    train_samples : int
        Total number of training samples.
    val_samples : int
        Total number of validation samples.
    test_samples : int
        Total number of test samples.

    results_folder : str
        Directory where training outputs, checkpoints, and plots are saved.
    file_training_csv : str
        Path to the CSV file for logging training/validation metrics.
    loss_png_file : str
        Path to save the training/validation loss plot.
    psnr_png_file : str
        Path to save the training/validation PSNR plot.
    final_model_pth_file : str
        File path to save the final model weights.
    training_log : str
        File path to save the training log (text-based).

    lr : float
        Learning rate for the optimizer.
    batch_size : int
        Batch size used during training.
    model_selection : str
        String identifier for the model architecture (used in file naming).
    epochs : int
        Number of total training epochs.
    """
    def __init__(
        self,
        model,
        ema_model,
        optimizer,
        compute_loss,
        loss_weights,
        device,

        train_loader,
        val_loader,
        test_loader,

        train_samples,
        val_samples,
        test_samples,

        results_folder,
        file_training_csv,
        loss_png_file,
        psnr_png_file,
        final_model_pth_file,
        training_log,

        lr,
        batch_size,
        model_selection,
        epochs,

        clipping = False
    ):
        self.model = model.to(device)
        self.ema_model = ema_model
        self.optimizer = optimizer
        self.compute_loss = compute_loss
        self.loss_weights = loss_weights
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        self.results_folder = results_folder
        self.file_training_csv = file_training_csv
        self.loss_png_file = loss_png_file
        self.psnr_png_file = psnr_png_file
        self.final_model_pth_file = final_model_pth_file
        self.best_model_path = None
        training_log = training_log


        # Configurar el logger global (esto se hace solo una vez)
        logging.basicConfig(
            filename=training_log,
            filemode='w',  # 'w' sobrescribe el archivo en cada ejecución. Usa 'a' para anexar.
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


        self.lr = lr
        self.batch_size = batch_size
        self.model_selection = model_selection
        self.epochs = epochs
        self.clipping = clipping

        self.train_loss = []
        self.train_psnr = []
        self.val_loss = []
        self.val_psnr = []



    def train_epoch(self):
        """
        Runs one full training epoch over the training dataset.

        Computes the average training loss and PSNR.

        Returns
        -------
        avg_loss : float
            Average loss across all training batches.
        avg_psnr : float
            Average PSNR across all training batches.
        """
        self.model.train()
        total_loss, total_psnr, total_samples = 0.0, 0.0, 0
        total_loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)
        num_batches = math.ceil(self.train_samples / self.batch_size)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = 0
            loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)  
            for j in range(len(self.compute_loss)):
                loss_j = self.loss_weights[j] * self.compute_loss[j](outputs, targets) 
                loss += loss_j
                loss_vec[j] = loss_j

            loss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            if self.ema_model is not None:
                self.ema_model.update()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            for j in range(len(self.compute_loss)):
                total_loss_vec[j] += loss_vec[j].item() * batch_size


            with torch.no_grad():
                batch_psnr = psnr(targets, outputs)
            total_psnr += batch_psnr * batch_size

            total_samples += batch_size

            # Console log (overwrites previous)
            formatted_losses = [f"{loss / total_samples:.4f}" for loss in total_loss_vec]
            print(
                f"Batch {batch_idx}/{num_batches} | "
                f"Batch PSNR: {batch_psnr:.2f} | "
                f"Total Loss: {total_loss / total_samples:.4f} | "
                f"Total Losses: {formatted_losses}",
                end='\r'
            )

            if batch_idx % 100 == 0:
                # Log to file
                logging.info(
                    f"Batch {batch_idx}/{num_batches} - "
                    f"Batch PSNR: {batch_psnr:.2f} - "
                    f"Avg Loss: {total_loss / total_samples:.4f} | "
                    f"Total Losses: {formatted_losses}"
                )

        print()  # Final clean line
        return total_loss / total_samples, total_psnr / total_samples, total_loss_vec/total_samples

    


    def validate_epoch(self, epoch):
        """
        Runs one validation epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number (for logging purposes).

        Returns
        -------
        avg_loss : float
            Average validation loss.
        avg_psnr : float
            Average validation PSNR.
        """
        self.ema_model.eval()
        total_loss, total_psnr = 0.0, 0.0
        total_loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.ema_model(inputs)
                loss =0
                loss_vec = np.zeros(len(self.compute_loss), dtype=np.float32)
                for j in range(len(self.compute_loss)):
                    loss_j = self.loss_weights[j] * self.compute_loss[j](outputs, targets) 
                    loss += loss_j
                    loss_vec[j]=loss_j
                batch_size = inputs.size(0)

                total_loss += loss.item() * batch_size
                total_psnr += psnr(targets, outputs) * batch_size
                for j in range(len(self.compute_loss)):
                    total_loss_vec[j] += loss_vec[j].item() * batch_size

        return total_loss / self.val_samples, total_psnr / self.val_samples, total_loss_vec/self.val_samples


    def save_checkpoint(self, epoch):
        """
        Saves the model and optimizer state as a checkpoint file.

        Parameters
        ----------
        epoch : int
            The current epoch number used in the filename.
        """
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        checkpoint = {
            "epoch": epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": self.train_loss[-1] if self.train_loss else None
        }
        path = os.path.join(
            self.results_folder,
            f"checkpoint_epoch_{epoch}_lr={self.lr}_batch_size={self.batch_size}_model={self.model_selection}.pth"
        )
        if os.path.exists(path):
            os.remove(path)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")


    def save_best_model(self, epoch):
        """
        Saves the model and optimizer state as a checkpoint file.

        Parameters
        ----------
        epoch : int
            The current epoch number used in the filename.
        """
        model_to_save = self.ema_model.module if hasattr(self.ema_model, "module") else self.ema_model
        checkpoint = {
            "epoch": epoch,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": self.train_loss[-1] if self.train_loss else None
        }
        path = os.path.join(
            self.results_folder,
            f"BestModel_epoch_{epoch}_lr={self.lr}_batch_size={self.batch_size}_model={self.model_selection}.pth"
        )
        self.best_model_path = "BestModel_epoch_{epoch}_lr={self.lr}_batch_size={self.batch_size}_model={self.model_selection}.pth"
        if os.path.exists(path):
            os.remove(path)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")



    def load_checkpoint(self, path):
        """
        Loads model and optimizer state from a given checkpoint file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.

        Returns
        -------
        start_epoch : int
            Epoch to resume training from.
        """
        checkpoint = torch.load(path, map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        loss = checkpoint.get("loss", None)
        print(f"Resumed training from epoch {start_epoch}, Previous Loss: {loss:.4f}")
        return start_epoch



    def save_training_log(self, train_loss, training_loss_vec, train_psnr, val_loss, val_loss_vec, val_psnr):
        """
        Appends training and validation metrics to a CSV log file.

        Parameters
        ----------
        train_loss : float
            Loss on the training set.
        training_loss_vec : np.ndarray
            Array of training losses.
        train_psnr : float
            PSNR on the training set.
        val_loss : float
            Loss on the validation set.
        val_loss_vec : np.ndarray
            Array of validation losses.
        val_psnr : float
            PSNR on the validation set.
        """
        with open(self.file_training_csv, mode="a", newline="") as f:
            writer = csv.writer(f)
            row = (
                [train_loss] +
                training_loss_vec.tolist() +
                [train_psnr, val_loss] +
                val_loss_vec.tolist() +
                [val_psnr]
            )
            writer.writerow(row)



    def plot_losses(self):
        """
        Plots and saves the training and validation loss (in dB) over all epochs.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(10 * np.log10(self.train_loss), color='orange', label='Train Loss')
        plt.plot(10 * np.log10(self.val_loss), color='red', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (dB)')
        plt.legend()
        if os.path.exists(self.loss_png_file):
            os.remove(self.loss_png_file)
        plt.savefig(self.loss_png_file)
        plt.show()


    def plot_psnr(self):
        """
        Plots and saves the training and validation PSNR over all epochs.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(self.train_psnr, color='green', label='Train PSNR (dB)')
        plt.plot(self.val_psnr, color='blue', label='Validation PSNR (dB)')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        if os.path.exists(self.psnr_png_file):
            os.remove(self.psnr_png_file)
        plt.savefig(self.psnr_png_file)
        plt.show()



    def save_final_model(self):
        """
        Saves the final model weights to the designated output path.
        """
        final_model = self.model.module if hasattr(self.model, "module") else self.model
        if os.path.exists(self.final_model_pth_file):
            os.remove(self.final_model_pth_file)
        torch.save(final_model.state_dict(), self.final_model_pth_file)



    def run(self, resume_checkpoint_path=None):
        """
        Executes the full training loop across all epochs.

        Supports optional checkpoint loading for resuming training.

        Parameters
        ----------
        resume_checkpoint_path : str, optional
            Path to a previously saved checkpoint file to resume training. If None, training starts from scratch.
        """

        start_epoch = 0
        best_val_psnr = float('-inf')  # Inicializa con el peor valor posible
        best_epoch = -1
        
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            start_epoch = self.load_checkpoint(resume_checkpoint_path)

        start = time.time()
        for epoch in range(start_epoch, self.epochs):
            print(f"\nEpoch {epoch + 1} of {self.epochs}")

            train_epoch_loss, train_epoch_psnr, train_epoch_loss_vec = self.train_epoch()
            val_epoch_loss, val_epoch_psnr, val_epoch_loss_vec = self.validate_epoch(epoch)

            print(f"Train PSNR: {train_epoch_psnr:.3f}")
            print(f"Val PSNR: {val_epoch_psnr:.3f}")

            # Log to file
            logging.info(
                f"\nEpoch {epoch + 1} of {self.epochs}- "
                f"Train PSNR: {train_epoch_psnr:.3f} - "
                f"Val PSNR: {val_epoch_psnr:.3f}"
            )


            self.train_loss.append(train_epoch_loss)
            self.train_psnr.append(train_epoch_psnr)
            self.val_loss.append(val_epoch_loss)
            self.val_psnr.append(val_epoch_psnr)

            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

            
            # Guarda el mejor modelo basado en PSNR de validación
            if val_epoch_psnr > best_val_psnr:
                best_val_psnr = val_epoch_psnr
                best_epoch = epoch
                self.save_best_model(epoch)

            self.save_training_log(train_epoch_loss,train_epoch_loss_vec, train_epoch_psnr, val_epoch_loss,val_epoch_loss_vec,val_epoch_psnr)

        end = time.time()
        print(f"\n✅ Finished training in: {(end - start) / 60:.2f} minutes")
        print(f"\n🏆 Best model was from epoch {best_epoch + 1} with Val PSNR: {best_val_psnr:.3f}")
        self.plot_losses()
        self.plot_psnr()
        print('\n💾 Saving final model...')
        self.save_final_model()

