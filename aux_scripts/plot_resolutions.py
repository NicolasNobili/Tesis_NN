"""
Figura — Comparativa de resoluciones espaciales (10 m, 30 m y 60 m)

Este script combina tres imágenes ya provistas (10 m, 30 m, 60 m) en una sola figura.
Configura las rutas al inicio del archivo.
"""

# ==========================
# CONFIGURACIÓN DEL USUARIO
# ==========================
IMG_10M       = "C:\\Users\\nnobi\\Downloads\\test_image10.webp"   # imagen 10 m
IMG_30M       = "C:\\Users\\nnobi\\Downloads\\test_image30.webp"   # imagen 30 m
IMG_60M       = "C:\\Users\\nnobi\\Downloads\\test_image60.webp"   # imagen 60 m
SALIDA_FIGURA = "figura_10_30_60.png"                              # salida
TITULO        = "Resoluciones espaciales: 10 m, 30 m y 60 m"
DPI           = 300
# ==========================

from PIL import Image
import matplotlib.pyplot as plt

def load_image(path):
    return Image.open(path).convert("RGB")

def main():
    im_10 = load_image(IMG_10M)
    im_30 = load_image(IMG_30M)
    im_60 = load_image(IMG_60M)

    # Solo resoluciones, sin nombres de satélites/bandas
    labels = ["10 m", "30 m", "60 m"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=DPI, constrained_layout=True)

    for ax, im, label in zip(axes, [im_10, im_30, im_60], labels):
        ax.imshow(im)
        ax.set_title(label, fontsize=12)
        ax.axis("off")

    # Título general más corto y más arriba para que no se superponga
    if TITULO:
        fig.suptitle(TITULO, fontsize=13, y=1.06)

    fig.savefig(SALIDA_FIGURA, dpi=DPI, bbox_inches="tight")
    print(f"[OK] Figura exportada a: {SALIDA_FIGURA}")

if __name__ == "__main__":
    main()
