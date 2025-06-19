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

from project_package.models.EDSR_model import EDSR
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.utils.train_common_routines import compute_loss_MSE
from project_package.utils.tester import Tester  # 👈 Clase personalizada

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
model_selection = 'EDSR_1806'
lr = 1e-5
batch_size = 32
dataset ='Dataset_Campo_patched_MatchedHist'
visualize_count = 20  # Número de ejemplos a visualizar

class EDSRConfig:
    def __init__(self, n_resblocks, n_feats, scale, n_colors, res_scale, rgb_range):
        self.n_resblocks = n_resblocks      # Número de bloques residuales
        self.n_feats = n_feats              # Número de características (features)
        self.scale = [scale]                # Escala de superresolución (lista con un elemento)
        self.n_colors = n_colors            # Número de canales (e.g. 3 para RGB)
        self.res_scale = res_scale          # Factor de escala residual
        self.rgb_range = rgb_range          # Rango de valores RGB (e.g. 255)

    def __repr__(self):
        return (f"EDSRConfig(n_resblocks={self.n_resblocks}, n_feats={self.n_feats}, "
                f"scale={self.scale}, n_colors={self.n_colors}, "
                f"res_scale={self.res_scale}, rgb_range={self.rgb_range})")
    
config = EDSRConfig(n_resblocks=16,n_feats=64,scale=2,n_colors=3,res_scale=0.1,rgb_range=1)

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
checkpoint_path = os.path.join(results_folder,'checkpoint_epoch_115_lr=0.0001_batch_size=32_model=EDSR.pth')
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
    model = EDSR(config)
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
