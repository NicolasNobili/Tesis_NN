import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import matplotlib.colors as mcolors

# --- Configuración base ---
base_path = "C:/Users/nnobi/Desktop/FIUBA/Tesis/Project/results_final"
output_dir = os.path.join(base_path, "figures", "20m_HCST_only")
os.makedirs(output_dir, exist_ok=True)

# Carpeta HCST
carpeta_hcst = "HybridTC"

# --- Iterar sobre samples ---
for sample_idx in range(1, 21):
    print(f"Procesando sample_{sample_idx}...")

    sample_name = f"sample_{sample_idx}"
    sample_dir_base = f"Test_Images_Dataset_Campo_20m_patched_MatchedHist/{sample_name}/originals"

    # Paths LR y HR
    img_lr_path = os.path.join(base_path, "EDSR", "20m", sample_dir_base, "input_LR.png")
    img_hr_path = os.path.join(base_path, "EDSR", "20m", sample_dir_base, "target_HR.png")

    if not (os.path.exists(img_lr_path) and os.path.exists(img_hr_path)):
        print(f"⚠️  No se encontraron imágenes LR/HR para {sample_name}. Se salta.")
        continue

    # Cargar LR / HR
    img_lr = np.array(Image.open(img_lr_path).convert("RGB"), dtype=np.float32)
    img_hr = np.array(Image.open(img_hr_path).convert("RGB"), dtype=np.float32)

    # Cargar SR (HCST)
    img_sr_path = os.path.join(base_path, carpeta_hcst, "20m", sample_dir_base, "output_SR.png")

    if not os.path.exists(img_sr_path):
        print(f"⚠️  No se encontró SR HCST para {sample_name}. Se salta.")
        continue

    img_sr = np.array(Image.open(img_sr_path).convert("RGB"), dtype=np.float32)

    # --- Diferencias ---
    diff_lr = np.abs(img_hr - img_lr) / 255.0
    diff_sr = np.abs(img_hr - img_sr) / 255.0

    diff_lr_mean = diff_lr.mean(axis=2)
    diff_sr_mean = diff_sr.mean(axis=2)

    # Escala común de diferencias
    diff_max = max(diff_lr_mean.max(), diff_sr_mean.max())
    norm = mcolors.Normalize(vmin=0, vmax=min(diff_max, 0.2))

    # --- Crear figura ---
    plt.figure(figsize=(14, 14))

    # === FILA 1: HR centrada ===
    plt.subplot(3, 2, (1, 2))
    plt.imshow(img_hr.astype(np.uint8))
    plt.title("HR (Ground Truth)")
    plt.axis("off")

    # === FILA 2: LR y SR ===
    plt.subplot(3, 2, 3)
    plt.imshow(img_lr.astype(np.uint8))
    plt.title("LR (Bicubic)")
    plt.axis("off")

    plt.subplot(3, 2, 4)
    plt.imshow(img_sr.astype(np.uint8))
    plt.title("HCST")
    plt.axis("off")

    # === FILA 3: |HR − LR| y |HR − SR| ===
    plt.subplot(3, 2, 5)
    plt.imshow(diff_lr_mean, cmap="magma", norm=norm)
    plt.title("|HR − LR|")
    plt.axis("off")

    plt.subplot(3, 2, 6)
    plt.imshow(diff_sr_mean, cmap="magma", norm=norm)
    plt.title("|HR − SR|")
    plt.axis("off")

    plt.tight_layout()

    # --- Guardar figura ---
    output_file = os.path.join(output_dir, f"hcst_sr_20m_{sample_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✔ Guardada figura: {output_file}")

print("\n🎉 Proceso completado. Figuras guardadas en:")
print(output_dir)
