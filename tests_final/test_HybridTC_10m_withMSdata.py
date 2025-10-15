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

from project_package.models.HybridTC_model import HybridTC, HybridTCConfig
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.loss_functions.gradient_variance_loss import GradientVariance
from project_package.utils.tester_MS_aux import Tester_MS_aux
from project_package.utils.utils import deserialize_losses

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Load Configuration from training_config.json
# ───────────────────────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))

model_selection = 'HybridTC_testMS'
low_res = '10m'

results_folder = os.path.join(project_dir, 'results_final', model_selection, low_res)
config_path = os.path.join(results_folder, 'training_config.json')

with open(config_path, 'r') as f:
    config_data = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Parámetros de configuración general
lr = config_data["lr"]
batch_size = config_data["batch_size"]
test_samples = config_data["test_samples"]
metadata_path = config_data["paths"]["metadata_path"]
checkpoint_path = os.path.join(results_folder, config_data["paths"]["best_model"])
visualize_count = 20

# Configuración del modelo
model_cfg = config_data["model_config"]
config = HybridTCConfig(**model_cfg)
losses, loss_weights = deserialize_losses(config_data=config_data, device=device)

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Patching config
# ───────────────────────────────────────────────────────────────────────────────
patching = True
if low_res == '10m':
    patch_size = {'low': (32, 32), 'high': (64, 64)}
    stride = {'low': (24, 24), 'high': (48, 48)}
elif low_res == '20m':
    patch_size = {'low': (16, 16), 'high': (64, 64)}
    stride = {'low': (12, 12), 'high': (48, 48)}

# ───────────────────────────────────────────────────────────────────────────────
# 📑 Lista de datasets a evaluar
# ───────────────────────────────────────────────────────────────────────────────
datasets_to_test = [
    "Dataset_Campo_10m_MS_patched_MatchedHist",
    "Dataset_Desierto_10m_MS_patched_MatchedHist",
    "Dataset_Selva_10m_MS_patched_MatchedHist",
    "Dataset_Montana_10m_MS_patched_MatchedHist"
]

multi_test_results_txt = os.path.join(results_folder, f"multi_test_results_model={model_selection}.txt")
with open(multi_test_results_txt, "w") as result_file:
    result_file.write(f"Resultados de evaluación para el modelo {model_selection} sobre múltiples datasets\n\n")

# ───────────────────────────────────────────────────────────────────────────────
# 🧪 Evaluación de múltiples datasets
# ───────────────────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
model = HybridTC(**vars(config))

for dataset in datasets_to_test:
    print(f"\nEvaluando dataset: {dataset}")

    dataset_folder = os.path.join(project_dir, 'datasets', dataset)
    dataset_test = PtWebDataset(
        os.path.join(dataset_folder, 'test.tar'),
        length=test_samples,
        batch_size=batch_size,
        shuffle_buffer=5 * batch_size,
        shuffle=False
    )
    dataloader_test = dataset_test.get_dataloader(num_workers=0)

    tester = Tester_MS_aux(
        model=model,
        device=device,
        compute_loss=losses,
        loss_weights=loss_weights,
        test_loader=dataloader_test,
        test_samples=test_samples,
        checkpoint_path=checkpoint_path,
        results_folder=results_folder,
        visualize_count=visualize_count,
        patching=True,
        patch_size=patch_size,
        stride=stride
    )

    avg_loss, avg_loss_vec, avg_psnr, avg_psnr_lr, avg_ssim, avg_lpips = tester.evaluate()

    with open(multi_test_results_txt, "a") as result_file:
        result_file.write(f"--- Dataset: {dataset} ---\n")
        result_file.write(f"Test Loss (MSE): {avg_loss:.6f}\n")
        result_file.write(f"Test PSNR: {avg_psnr:.2f} dB\n")
        result_file.write(f"Bicubic PSNR: {avg_psnr_lr:.2f} dB\n")
        result_file.write(f"Test SSIM: {avg_ssim:.6f}\n")
        result_file.write(f"Test LPIPS: {avg_lpips:.6f}\n\n")

    tester.visualize_results(folder_path=os.path.join(results_folder,'Test_Images_' + dataset))