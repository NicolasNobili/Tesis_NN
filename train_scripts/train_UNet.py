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
from torch import nn

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils import train_common_routines as tcr
from project_package.models.UNet_model import UNet1,UNetConfig 
from project_package.dataset_manager.webdataset_dataset import PtWebDataset
from project_package.utils.trainer import Trainer 
from project_package.utils.trainer_with_ema import Trainer_EMA
from project_package.loss_functions.gradient_variance_loss import GradientVariance 
from project_package.loss_functions.edge_loss import EdgeLossRGB
from project_package.utils.utils import serialize_losses

# ───────────────────────────────────────────────────────────────────────────────
# 🔧 Configuration
# ───────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_selection = 'UNet_2108'
epochs = 200
lr = 3e-4
batch_size = 32
dataset = 'Dataset_Campo_10m_patched_MatchedHist_InputMatch' 
low_res = '10m'
losses = [nn.MSELoss() ,EdgeLossRGB().to(device)]
losses_weights = [1,0.1]

config = UNetConfig(scale=2,n_channels=[32,64,128,256],n_colors=3,rgb_range=1)

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
# 💾 Guardar Configuración de Entrenamiento en JSON
# ───────────────────────────────────────────────────────────────────────────────

# Serializar losses
# Convertimos la lista de losses en una forma serializable
losses_serializable = serialize_losses(losses=losses,losses_weights=losses_weights)

# Crear configuración
training_config = {
    "model_selection": model_selection,
    "epochs": epochs,
    "lr": lr,
    "batch_size": batch_size,
    "dataset": dataset,
    "low_res": low_res,
    "device": str(device),
    "losses": losses_serializable,
    "model_config": {
        "scale": config.scale,
        "n_channels": config.n_channels,
        "n_colors":config.n_colors,
        "rgb_range":config.rgb_range
    },
    "train_samples": train_samples,
    "val_samples": val_samples,
    "test_samples": test_samples,
    "paths": {
        "results_folder": results_folder,
        "loss_png_file": loss_png_file,
        "psnr_png_file": psnr_png_file,
        "final_model_pth_file": final_model_pth_file,
        "file_training_csv": file_training_csv,
        "training_log": training_log,
        "metadata_path": metadata_path
    }
}

# Guardar archivo JSON
config_json_path = os.path.join(results_folder, "training_config.json")
with open(config_json_path, 'w') as f:
    json.dump(training_config, f, indent=4)

print(f"✔️ Configuración guardada en: {config_json_path}")

# ───────────────────────────────────────────────────────────────────────────────
# 🚀 Training Pipeline
# ───────────────────────────────────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True

model = UNet1(config).to(device)
model.apply(tcr.init_small)
#ema_model = tcr.EMA(model, decay=0.999)

print("The model:")
print(model)

model.count_parameters()
print(f"Total Parameters: {model.total_params:,}")
print(f"Trainable Parameters: {model.trainable_params:,}")

model = tcr.multi_GPU_training(model)

#Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay=1e-4)

# Cosine Scheduler + warmup
warmup_epochs = 5
warmup = optim.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
cosine = optim.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
scheduler = optim.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

# Datasets
dataset_train = PtWebDataset(os.path.join(dataset_folder, 'train-*.tar'), length=train_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
dataset_val = PtWebDataset(os.path.join(dataset_folder, 'val-*.tar'), length=val_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)
dataset_test = PtWebDataset(os.path.join(dataset_folder, 'test.tar'), length=test_samples, batch_size=batch_size, shuffle_buffer=5 * batch_size)

dataloader_train = dataset_train.get_dataloader(num_workers=6)
dataloader_val = dataset_val.get_dataloader(num_workers=2)
dataloader_test = dataset_test.get_dataloader(num_workers=0)


# Entrenador
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler= scheduler,
    compute_loss = losses ,
    loss_weights = losses_weights,
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

# Agregar checkpoint final al JSON
training_config["paths"]["best_model"] = trainer.best_model_path 

# Reescribir JSON actualizado
with open(config_json_path, 'w') as f:
    json.dump(training_config, f, indent=4, weight_decay=1e-4)

# ───────────────────────────────────────────────────────────────────────────────
# 🔄 Actualizar JSON con checkpoint final (si existe)
# ───────────────────────────────────────────────────────────────────────────────

def get_latest_checkpoint(folder):
    files = [f for f in os.listdir(folder) if f.startswith('checkpoint_epoch') and f.endswith('.pth')]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split('_')[2]))  # checkpoint_epoch_XX.pth
    return os.path.join(folder, files[-1])

latest_checkpoint = get_latest_checkpoint(results_folder)
training_config["paths"]["final_model_checkpoint"] = latest_checkpoint if latest_checkpoint else final_model_pth_file

# Reescribir JSON actualizado
with open(config_json_path, 'w') as f:
    json.dump(training_config, f, indent=4)

print(f"📦 JSON actualizado con checkpoint final: {training_config['paths']['final_model_checkpoint']}")

