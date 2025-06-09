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
# Add custom project folder to system path to enable local module imports
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils.train_common_routines import psnr


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        compute_loss,
        device,
        train_loader,
        val_loader,
        test_loader,
        train_samples,
        val_samples,
        test_samples,
        results_folder,
        file_training_losses,
        loss_png_file,
        psnr_png_file,
        final_model_pth_file,
        lr,
        batch_size,
        model_selection,
        epochs,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.compute_loss = compute_loss
        self.device = device

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        self.results_folder = results_folder
        self.file_training_losses = file_training_losses
        self.loss_png_file = loss_png_file
        self.psnr_png_file = psnr_png_file
        self.final_model_pth_file = final_model_pth_file

        self.lr = lr
        self.batch_size = batch_size
        self.model_selection = model_selection
        self.epochs = epochs

        self.train_loss = []
        self.train_psnr = []
        self.val_loss = []
        self.val_psnr = []



    def train_epoch(self):
        self.model.train()
        total_loss, total_psnr, total_samples = 0.0, 0.0, 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size

            with torch.no_grad():
                batch_psnr = psnr(targets, outputs)
            total_psnr += batch_psnr * batch_size
            total_samples += batch_size

        return total_loss / total_samples, total_psnr / total_samples
    


    def validate_epoch(self, epoch):
        self.model.eval()
        total_loss, total_psnr = 0.0, 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, targets)
                batch_size = inputs.size(0)

                total_loss += loss.item() * batch_size
                total_psnr += psnr(targets, outputs) * batch_size

        return total_loss / self.val_samples, total_psnr / self.val_samples



    def save_checkpoint(self, epoch):
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



    def load_checkpoint(self, path):
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



    def save_training_log(self, train_loss, train_psnr, val_loss, val_psnr):
        with open(self.file_training_losses, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([train_loss, train_psnr, val_loss, val_psnr])



    def plot_losses(self):
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
        final_model = self.model.module if hasattr(self.model, "module") else self.model
        if os.path.exists(self.final_model_pth_file):
            os.remove(self.final_model_pth_file)
        torch.save(final_model.state_dict(), self.final_model_pth_file)



    def run(self, resume_checkpoint_path=None):
        start_epoch = 0
        if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
            start_epoch = self.load_checkpoint(resume_checkpoint_path)

        start = time.time()
        for epoch in range(start_epoch, self.epochs):
            print(f"\nEpoch {epoch + 1} of {self.epochs}")

            train_epoch_loss, train_epoch_psnr = self.train_epoch()
            val_epoch_loss, val_epoch_psnr = self.validate_epoch(epoch)

            print(f"Train PSNR: {train_epoch_psnr:.3f}")
            print(f"Val PSNR: {val_epoch_psnr:.3f}")

            self.train_loss.append(train_epoch_loss)
            self.train_psnr.append(train_epoch_psnr)
            self.val_loss.append(val_epoch_loss)
            self.val_psnr.append(val_epoch_psnr)

            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

            self.save_training_log(train_epoch_loss, train_epoch_psnr, val_epoch_loss, val_epoch_psnr)

        end = time.time()
        print(f"\n✅ Finished training in: {(end - start) / 60:.2f} minutes")
        self.plot_losses()
        self.plot_psnr()
        print('\n💾 Saving final model...')
        self.save_final_model()
