import subprocess
import sys
import os

# Obtener la ruta absoluta de la carpeta donde está este script
carpeta = os.path.dirname(os.path.abspath(__file__))

print(f"Buscando scripts en: {carpeta}\n")

# Buscar todos los archivos .py dentro de la carpeta (excluyendo este mismo archivo)
scripts = [
    f for f in os.listdir(carpeta)
    if f.endswith(".py") and f != os.path.basename(__file__)
]

# Ejecutar cada script encontrado
for s in sorted(scripts):
    ruta = os.path.join(carpeta, s)
    print(f"Ejecutando {ruta}...\n")
    subprocess.run([sys.executable, ruta])
    print(f"Finalizado {s}\n{'-'*60}\n")
