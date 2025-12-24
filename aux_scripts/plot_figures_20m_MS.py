import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import matplotlib.colors as mcolors

# --- Configuración base ---
base_path = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final"
output_dir = os.path.join(base_path, "figures", "20m_MS")
os.makedirs(output_dir, exist_ok=True)

# Lista de arquitecturas y carpetas
arquitecturas = {
    "RCRCA (RGB)": "RCAN_testMS",
    "RCRCA": "RCAN_MS",
    "HCST (RGB)": "HybridTC_testMS",
    "HCST": "HybridTC_MS"
}

# Diccionario con PSNRs
psnr_dict = {
    "RCRCA (RGB)":     [0,0,0,0,0,0,0,0,0,0,33.73,0,0,0,0,36.10,0,0,0,0],
    "RCRCA":           [0,0,0,0,0,0,0,0,0,0,33.87,0,0,0,0,35.88,0,0,0,0],
    "HCST (RGB)":      [0,0,0,0,0,0,0,0,0,0,34.13,0,0,0,0,36.42,0,0,0,0],
    "HCST":            [0,0,0,0,0,0,0,0,0,0,34.11,0,0,0,0,36.63,0,0,0,0]
}

# --- Iterar sobre todas las samples ---
for sample_idx in range(1, 21):
    print(f"Procesando sample_{sample_idx}...")

    sample_name = f"sample_{sample_idx}"
    sample_dir_base = f"Test_Images_Dataset_Campo_20m_MS_patched_MatchedHist/{sample_name}/originals"

    # Paths de LR y HR 
    img_lr_path = os.path.join(base_path, "RCAN_MS", "20m", sample_dir_base, "input_LR.png")
    img_hr_path = os.path.join(base_path, "RCAN_MS", "20m", sample_dir_base, "target_HR.png")

    if not (os.path.exists(img_lr_path) and os.path.exists(img_hr_path)):
        print(f"  No se encontraron imágenes LR/HR para {sample_name}. Se salta.")
        continue

    # Cargar imágenes base
    img_lr = np.array(Image.open(img_lr_path).convert("RGB"), dtype=np.float32)
    img_hr = np.array(Image.open(img_hr_path).convert("RGB"), dtype=np.float32)

    # Preparar imágenes SR con sus PSNR
    imagenes_sr = {}
    for nombre, carpeta in arquitecturas.items():
        sr_path = os.path.join(base_path, carpeta, "20m", sample_dir_base, "output_SR.png")
        if os.path.exists(sr_path):
            psnr_value = psnr_dict[nombre][sample_idx - 1]
            imagenes_sr[nombre] = {"path": sr_path, "psnr": psnr_value}
        else:
            print(f" No se encontró SR para {nombre} ({sample_name})")

    num_sr = len(imagenes_sr)
    cols = 4
    rows_sr = int(np.ceil(num_sr / 2))
    rows_total = 1 + rows_sr

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

    # --- Calcular escala global para normalización ---
    sr_items = list(imagenes_sr.items())
    diff_max_global = 0
    for _, datos in sr_items:
        img_sr = np.array(Image.open(datos["path"]).convert("RGB"), dtype=np.float32)
        diff = np.abs(img_hr - img_sr) / 255.0
        diff_max_global = max(diff_max_global, diff.mean(axis=2).max())

    # Escala fija (0 a diff_max_global, o limitar a 0.2 para contraste razonable)
    norm = mcolors.Normalize(vmin=0, vmax=min(diff_max_global, 0.2))

    # --- Filas siguientes: pares SR / diferencia ---
    for idx, (nombre, datos) in enumerate(sr_items):
        img_sr = np.array(Image.open(datos["path"]).convert("RGB"), dtype=np.float32)
        diff = np.abs(img_hr - img_sr) / 255.0  # normalizado a [0,1]
        diff_mean = diff.mean(axis=2)

        fila = idx // 2 + 1
        col_offset = (idx % 2) * 2
        base_index = fila * cols + col_offset + 1

        # Imagen SR
        plt.subplot(rows_total, cols, base_index)
        plt.imshow(img_sr.astype(np.uint8))
        psnr_text = f"PSNR: {datos['psnr']:.2f} dB" if datos["psnr"] else "PSNR: --"
        plt.title(f"{nombre}\n{psnr_text}")
        plt.axis("off")

        # Diferencia (escala fija)
        plt.subplot(rows_total, cols, base_index + 1)
        plt.imshow(diff_mean, cmap="magma", norm=norm)
        plt.title("|HR − SR|")
        plt.axis("off")

    plt.tight_layout()

    # --- Guardar imagen ---
    output_file = os.path.join(output_dir, f"comparacion_sr_20m_{sample_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Guardada figura: {output_file}")

print("\n Proceso completado. Todas las figuras guardadas en:")
print(output_dir)
