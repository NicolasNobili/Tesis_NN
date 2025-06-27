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

from project_package.models.RCAN_model import RCAN
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.utils.train_common_routines import compute_loss_MSE
from project_package.utils.tester import Tester  # 👈 Clase personalizada

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
model_selection = 'RCAN_2006'
epochs = 200
lr = 1e-4
batch_size = 32
dataset = 'Dataset_Campo_10m_patched_MatchedHist'   
visualize_count = 20  # Número de ejemplos a visualizar

class RCANConfig:
    def __init__(self, scale, num_features, num_rg, num_rcab, reduction, upscaling):
        self.scale = scale
        self.num_features = num_features
        self.num_rg = num_rg
        self.num_rcab = num_rcab
        self.reduction = reduction
        self.upscaling = upscaling

    def __repr__(self):
        return (f"ModelConfig(scale={self.scale}, num_features={self.num_features}, "
                f"num_rg={self.num_rg}, num_rcab={self.num_rcab}, "
                f"reduction={self.reduction}, upscaling={self.upscaling})")
    

config = RCANConfig(scale=2 , num_features=64 ,num_rg=4, num_rcab=8, reduction=16 , upscaling=True)

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
#checkpoint_path = os.path.join(results_folder, f"model_lr=0.0001_batch_size=32_model=EDSR.pth")
checkpoint_path = os.path.join(results_folder,'checkpoint_epoch_155_lr=0.0001_batch_size=32_model=RCAN.pth')
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
    model = RCAN(config)
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
