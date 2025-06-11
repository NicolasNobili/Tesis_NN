# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import torch

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.models.ConvNet_model import SRCNN_large
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.utils.train_common_routines import compute_loss_MSE
from project_package.utils.tester import Tester  # 👈 Clase personalizada

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
model_selection = 'SRCNN_large'
lr = 1e-5
batch_size = 32
dataset = 'dataset_campo'
visualize_count = 5  # Número de ejemplos a visualizar

# ───────────────────────────────────────────────────────────────────────────────
# 📁 Paths Setup
# ───────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))

dataset_folder = os.path.join(project_dir, 'datasets', dataset)
metadata_path = os.path.join(dataset_folder, 'metadata.json')

with open(metadata_path, "r") as f:
    metadata = json.load(f)

test_samples = metadata["splits"]["test"]["num_samples"]

results_folder = os.path.join(project_dir, 'results', model_selection)
# checkpoint_path = os.path.join(results_folder, f"model_lr={lr}_batch_size={batch_size}_model={model_selection}.pth")
checkpoint_path = os.path.join(results_folder,'checkpoint_epoch_195_lr=1e-05_batch_size=32_model=SRCNN_large.pth')
test_results_txt = os.path.join(results_folder, f"test_results_lr={lr}_batch_size={batch_size}_model={model_selection}.txt")

# ───────────────────────────────────────────────────────────────────────────────
# 🧪 Test Pipeline
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    torch.backends.cudnn.benchmark = True

    # Datasets
    dataset_test = PtWebDataset(os.path.join(dataset_folder, 'test.tar'), length=test_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
    dataloader_test = dataset_test.get_dataloader(num_workers=0)

    # Model setup
    model = SRCNN_large()
    tester = Tester(
        model=model,
        device=device,
        compute_loss=compute_loss_MSE,
        test_loader=dataloader_test,
        test_samples=test_samples,
        checkpoint_path=checkpoint_path,
        results_folder=results_folder,
        visualize_count=visualize_count
    )

    # Run evaluation
    avg_loss, avg_psnr = tester.evaluate()

    # Save test results
    with open(test_results_txt, "w") as f:
        f.write(f"Test Loss (MSE): {avg_loss:.6f}\n")
        f.write(f"Test PSNR: {avg_psnr:.2f} dB\n")

    # Visualize predictions
    tester.visualize_results()
