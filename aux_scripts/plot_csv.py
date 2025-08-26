import pandas as pd
import matplotlib.pyplot as plt

def graficar_columnas_csv(ruta_csv):
    # Leer el CSV en un DataFrame
    df = pd.read_csv(ruta_csv)
    
    # Iterar por cada columna con índice
    for i, columna in enumerate(df.columns, start=1):
        plt.figure()  # Crear un gráfico nuevo para cada columna
        plt.plot(df.index, df[columna])
        plt.title(f"Columna {i}")  # Usar número de columna
        plt.xlabel("Número de muestra (n)")
        plt.ylabel(columna)  # Nombre real de la columna como etiqueta Y
        plt.grid(True)
        plt.show()


# Ejemplo de uso
graficar_columnas_csv("C:\\Users\\nnobi\\Desktop\\FIUBA\\Tesis\\Project\\results\\UNet_2108\\10m\\training_losses_lr=0.0003_batch_size=32_model=UNet_2108.csv")
