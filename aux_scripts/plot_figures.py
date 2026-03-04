import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Diccionario con las imágenes SR y sus PSNR
imagenes_sr = {
    "RC": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/SRCNN_small/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 32.92},
    "RCR": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/RCR/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 32.95},
    "RCRCA": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/RCRAC/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 33.94},
    "UNet": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/UNet/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 33.28},
    "ST": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/Transformer/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 34.26},
    "HCST": {"path": "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/HybridTC/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/output_SR.png", "psnr": 34.49}
}

# Paths base
img_lr_path = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/RCR/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/input_LR.png"
img_hr_path = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/RCR/20m/Test_Images_Dataset_Campo_20m_patched_MatchedHist/sample_6/originals/target_HR.png"

# Cargar imágenes
img_lr = np.array(Image.open(img_lr_path).convert("RGB"), dtype=np.float32)
img_hr = np.array(Image.open(img_hr_path).convert("RGB"), dtype=np.float32)

# Parámetros de figura
num_sr = len(imagenes_sr)
cols = 4  # SR, diff, SR, diff
rows_sr = int(np.ceil(num_sr / 2))  # dos pares por fila
rows_total = 1 + rows_sr  # +1 para la fila LR/HR

plt.figure(figsize=(16, 4 * rows_total))

# --- Fila superior: LR y HR ---
plt.subplot(rows_total, 2, 1)
plt.imshow(img_lr.astype(np.uint8))
plt.title("LR (Bicubic)")
plt.axis("off")

plt.subplot(rows_total, 2, 2)
plt.imshow(img_hr.astype(np.uint8))
plt.title("HR (Ground Truth)")
plt.axis("off")

# --- Filas siguientes: pares SR / diferencia ---
sr_items = list(imagenes_sr.items())
for idx, (nombre, datos) in enumerate(sr_items):
    img_sr = np.array(Image.open(datos["path"]).convert("RGB"), dtype=np.float32)
    diff = np.abs(img_hr - img_sr)
    diff = diff / diff.max()  # normalizar para visualización
    diff_img = (diff * 255).astype(np.uint8)

    fila = idx // 2 + 1  # fila después de la primera
    col_offset = (idx % 2) * 2  # 0 o 2 (para el par izquierdo o derecho)
    base_index = fila * cols + col_offset + 1  # índice del subplot

    # Imagen SR
    plt.subplot(rows_total, cols, base_index)
    plt.imshow(img_sr.astype(np.uint8))
    plt.title(f"{nombre}\nPSNR: {datos['psnr']:.2f} dB")
    plt.axis("off")

    # Diferencia
    plt.subplot(rows_total, cols, base_index + 1)
    plt.imshow(diff_img, cmap="hot")
    plt.title("Diferencia |HR - SR|")
    plt.axis("off")

plt.tight_layout()

# --- Guardar ---
carpeta_salida = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final/figures"
os.makedirs(carpeta_salida, exist_ok=True)
salida = os.path.join(carpeta_salida, "comparacion_sr_20m_s6_diff.png")

plt.savefig(salida, dpi=300, bbox_inches="tight")
plt.close()

print(f"Imagen guardada en:\n{salida}")
