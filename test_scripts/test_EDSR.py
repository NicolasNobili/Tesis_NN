# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import torch
from torch import nn

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.models.EDSR_model import EDSR,EDSRConfig
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.loss_functions.gradient_variance_loss import GradientVariance
from project_package.utils.tester import Tester 

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Load Configuration from training_config.json
# ───────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))

model_selection = 'EDSR_1806'
low_res = '10m'

results_folder = os.path.join(project_dir, 'results', model_selection, low_res)
config_path = os.path.join(results_folder, 'training_config.json')

with open(config_path, 'r') as f:
    config_data = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Parámetros
lr = config_data["lr"]
batch_size = config_data["batch_size"]
dataset = config_data["dataset"]
test_samples = config_data["test_samples"]
metadata_path = config_data["paths"]["metadata_path"]
checkpoint_path = os.path.join(results_folder, 'checkpoint_epoch_195_lr=0.0001_batch_size=32_model=EDSR.pth')  # Puedes hacer esto dinámico si lo prefieres
test_results_txt = os.path.join(results_folder, f"test_results_lr={lr}_batch_size={batch_size}_model={model_selection}.txt")
visualize_count = 20

# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Model Config Reconstruction
# ───────────────────────────────────────────────────────────────────────────────
model_cfg = config_data["model_config"]
config = EDSRConfig(**model_cfg)

# ───────────────────────────────────────────────────────────────────────────────
# 🔁 Loss Functions Reconstruction
# ───────────────────────────────────────────────────────────────────────────────
losses, loss_weights = deserialize_losses(config_data=config_data,device=device)

# ───────────────────────────────────────────────────────────────────────────────
# 📁 Paths y Dataset
# ───────────────────────────────────────────────────────────────────────────────
dataset_folder = os.path.join(project_dir, 'datasets', dataset)
dataset_test = PtWebDataset(os.path.join(dataset_folder, 'test.tar'), length=test_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
dataloader_test = dataset_test.get_dataloader(num_workers=0)

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Patches
# ───────────────────────────────────────────────────────────────────────────────
patching = True
if (low_res == '10m'):
    patch_size = {'low':(32,32), 'high':(64,64)}
    stride = {'low':(24,24), 'high':(48,48)}
elif (low_res == '20m'):
    patch_size = {'low':(16,16), 'high':(64,64)}
    stride = {'low':(12,12), 'high':(48,48)}

# ───────────────────────────────────────────────────────────────────────────────
# 🧪 Test Pipeline
# ───────────────────────────────────────────────────────────────────────────────
print("Device:", device)
torch.backends.cudnn.benchmark = True

model = EDSR(config)

tester = Tester(
    model=model,
    device=device,
    compute_loss=losses,
    loss_weights=loss_weights,
    test_loader=dataloader_test,
    test_samples=test_samples,
    checkpoint_path=checkpoint_path,
    results_folder=results_folder,
    visualize_count=visualize_count,
    patching=False,  # Puedes poner `True` si estás evaluando con parches
    patch_size=patch_size,
    stride=stride
)

# Run evaluation
avg_loss, avg_loss_vec, avg_psnr, avg_psnr_lr, avg_ssim, avg_lpips = tester.evaluate()

# Save test results
with open(test_results_txt, "w") as f:
    f.write(f"Test Loss (MSE): {avg_loss:.6f}\n")
    f.write(f"Test PSNR: {avg_psnr:.2f} dB\n")
    f.write(f"Bicubic PSNR: {avg_psnr_lr:.2f} dB\n")
    f.write(f"Test SSIM: {avg_ssim:.6f}\n")
    f.write(f"Test LPIPS: {avg_lpips:.6f}\n")

# Visualize predictions
tester.visualize_results()
