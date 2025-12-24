import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import matplotlib.colors as mcolors

# --- Configuración base ---
base_path = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final"
output_dir = os.path.join(base_path, "figures", "20m_MS_HCST_only_MS")
os.makedirs(output_dir, exist_ok=True)

# Arquitecturas MS
arquitecturas = {
    "HCST": "HybridTC_MS"
}

# --- Iterar samples ---
for sample_idx in range(1, 21):
    print(f"Procesando sample_{sample_idx}...")

    sample_name = f"sample_{sample_idx}"
    sample_dir = f"Test_Images_Dataset_Campo_20m_MS_patched_MatchedHist/{sample_name}/originals"

    # Cargar LR y HR desde RCAN_MS
    img_lr_path = os.path.join(base_path, "RCAN_MS", "20m", sample_dir, "input_LR.png")
    img_hr_path = os.path.join(base_path, "RCAN_MS", "20m", sample_dir, "target_HR.png")

    if not (os.path.exists(img_lr_path) and os.path.exists(img_hr_path)):
        print("⚠️  No LR/HR, skip.")
        continue

    img_lr = np.array(Image.open(img_lr_path).convert("RGB"), dtype=np.float32)
    img_hr = np.array(Image.open(img_hr_path).convert("RGB"), dtype=np.float32)

    # Cargar SR modelos MS
    imagenes_sr = {}
    for nombre, carpeta in arquitecturas.items():
        sr_path = os.path.join(base_path, carpeta, "20m", sample_dir, "output_SR.png")
        if os.path.exists(sr_path):
            imagenes_sr[nombre] = np.array(Image.open(sr_path).convert("RGB"), dtype=np.float32)
        else:
            print(f"⚠️ No SR para {nombre} en {sample_name}")

    if len(imagenes_sr) == 0:
        continue

    # --- Diferencias ---
    diff_lr = np.abs(img_hr - img_lr) / 255.0
    diff_lr_mean = diff_lr.mean(axis=2)

    diff_sr = {}
    for nombre, img_sr in imagenes_sr.items():
        diff = np.abs(img_hr - img_sr) / 255.0
        diff_sr[nombre] = diff.mean(axis=2)

    # Normalización común para diferencias
    diff_max = max(diff_lr_mean.max(), *(v.max() for v in diff_sr.values()))
    norm = mcolors.Normalize(vmin=0, vmax=min(diff_max, 0.2))

    # === Crear figura ===
    plt.figure(figsize=(12, 15))

    # === FILA 1: HR ===
    plt.subplot(4, 2, (1, 2))
    plt.imshow(img_hr.astype(np.uint8))
    plt.title("HR (Ground Truth)")
    plt.axis("off")

    # === FILA 2: LR ===
    plt.subplot(4, 2, 3)
    plt.imshow(img_lr.astype(np.uint8))
    plt.title("LR (Bicubic)")
    plt.axis("off")

    # === FILA 2: HCST ===
    plt.subplot(4, 2, 4)
    if "HCST" in imagenes_sr:
        plt.imshow(imagenes_sr["HCST"].astype(np.uint8))
        plt.title("HCST (MS)")
    else:
        plt.text(0.5, 0.5, "No HCST", ha="center", va="center")
    plt.axis("off")

    # === FILA 3: |HR − LR| ===
    plt.subplot(4, 2, 5)
    plt.imshow(diff_lr_mean, cmap="magma", norm=norm)
    plt.title("|HR − LR|")
    plt.axis("off")

    # === FILA 3: |HR − HCST| ===
    plt.subplot(4, 2, 6)
    if "HCST" in diff_sr:
        plt.imshow(diff_sr["HCST"], cmap="magma", norm=norm)
        plt.title("|HR − HCST|")
    else:
        plt.text(0.5, 0.5, "No HCST", ha="center", va="center")
    plt.axis("off")

    plt.tight_layout()

    # Guardar
    output_file = os.path.join(output_dir, f"comparacion_MS_20m_{sample_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ Guardado: {output_file}")

print("\n🎉 Proceso completado. Figuras guardadas en:")
print(output_dir)
