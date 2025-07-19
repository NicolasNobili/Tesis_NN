import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
csv_file = "datos.csv"  # Cambia esto al nombre de tu archivo CSV
delimiter = ","         # Usa ";" si es CSV con punto y coma
use_headers = True      # Si la primera fila tiene nombres de columnas

# Títulos personalizados para cada gráfico (uno por columna)
custom_titles = [
    "Gráfico 1",
    "Gráfico 2",
    "Gráfico 3",
    # Agrega más si hay más columnas
]

# === LECTURA DEL CSV ===
df = pd.read_csv(csv_file, delimiter=delimiter, header=0 if use_headers else None)

# === PLOTEO DE CADA COLUMNA ===
for i, col in enumerate(df.columns):
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df[col], marker='o')
    
    # Título personalizado si está disponible, si no usa el nombre de la columna
    title = custom_titles[i] if i < len(custom_titles) else f"Columna {col}"
    plt.title(title)
    
    plt.xlabel("Índice")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.tight_layout()
    
    # Guarda la imagen (opcional)
    plt.savefig(f"grafico_columna_{i+1}.png")
    plt.close()

print("✅ Gráficos generados y guardados como PNG.")
