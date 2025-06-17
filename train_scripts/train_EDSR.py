# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import sys
import json

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Scientific & Data Libraries
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.optim as optim

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils import train_common_routines as tcr
from project_package.models.EDSR_model import EDSR
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.utils.trainer import Trainer  # Asegúrate de importar tu clase Trainer

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
model_selection = 'EDSR'
epochs = 200
lr = 1e-4
batch_size = 32
dataset = 'Dataset_Campo_patched'

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

train_samples = metadata["splits"]["train"]["num_samples"]
val_samples = metadata["splits"]["val"]["num_samples"]
test_samples = metadata["splits"]["test"]["num_samples"]

results_folder = os.path.join(project_dir, 'results', model_selection)
os.makedirs(results_folder, exist_ok=True)


loss_png_file = os.path.join(results_folder, f"loss_lr={lr}_batch_size={batch_size}_model={model_selection}.png")
psnr_png_file = os.path.join(results_folder, f"psnr_lr={lr}_batch_size={batch_size}_model={model_selection}.png")
final_model_pth_file = os.path.join(results_folder, f"model_lr={lr}_batch_size={batch_size}_model={model_selection}.pth")
file_training_csv = os.path.join(results_folder, f"training_losses_lr={lr}_batch_size={batch_size}_model={model_selection}.csv")
training_log = os.path.join(results_folder,f"log_lr={lr}_batch_size={batch_size}_model={model_selection}.log")


# ───────────────────────────────────────────────────────────────────────────────
# 🚀 Training Pipeline
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    torch.backends.cudnn.benchmark = True

    model = EDSR(config).to(device)
    print("The model:")
    print(model)

    model.count_parameters()
    print(f"Total Parameters: {model.total_params:,}")
    print(f"Trainable Parameters: {model.trainable_params:,}")

    model = tcr.multi_GPU_training(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Datasets
    dataset_train = PtWebDataset(os.path.join(dataset_folder, 'train-*.tar'), length=train_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
    dataset_val = PtWebDataset(os.path.join(dataset_folder, 'val-*.tar'), length=val_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
    dataset_test = PtWebDataset(os.path.join(dataset_folder, 'test.tar'), length=test_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)

    dataloader_train = dataset_train.get_dataloader(num_workers=0)
    dataloader_val = dataset_val.get_dataloader(num_workers=0)
    dataloader_test = dataset_test.get_dataloader(num_workers=0)

    # Entrenador
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        compute_loss=tcr.compute_loss_MSE,
        device=device,

        train_loader=dataloader_train,
        val_loader=dataloader_val,
        test_loader=dataloader_test,

        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,

        results_folder=results_folder,
        file_training_csv=file_training_csv,
        loss_png_file=loss_png_file,
        psnr_png_file=psnr_png_file,
        final_model_pth_file=final_model_pth_file,
        training_log=training_log,

        lr=lr,
        batch_size=batch_size,
        model_selection=model_selection,
        epochs=epochs
    )

    # 🚀 Ejecutar entrenamiento completo
    trainer.run()  # Puedes pasar un path con resume_checkpoint_path='...' si deseas reanudar
