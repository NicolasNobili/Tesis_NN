# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import sys
import io
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch
import geopandas as gpd

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────

# Añadir ruta al proyecto para importar módulos personalizados
if os.name == "posix":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
else:
    sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

from project_package.utils import train_common_routines2 as tcr
from project_package.data_processing import sen2venus_routines as s2v
from project_package.utils import utils as utils

# ───────────────────────────────────────────────────────────────────────────────
# 🏁 Main execution
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Obtener ruta del script y del proyecto
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))

    # Ruta a los datos de entrada Sen2Venus
    s2v_filtered_path = (
        'C:/Users/nnobi/Desktop/FIUBA/Tesis/Sen2Venus_rgb'
        if os.name == "nt"
        else '/media/nicolasn/New Volume/Sen2Venus_rgb'
    )

    # Directorio de salida donde se generará el dataset
    output_path = os.path.join(project_dir, 'datasets', 'dataset_test1')
    # output_path = os.path.join(project_dir, 'datasets')
    os.makedirs(output_path, exist_ok=True)  # Crear si no existe

    # Sitios a procesar (podés agregar más si querés)
    selected_sites = ['ALSACE','ANJI','BENGA']

    # Generar dataset en formato .tar usando WebDataset
    counts = s2v.generate_dataset_tar_split2(
        dir_sen2venus_path=s2v_filtered_path,
        sites=selected_sites,
        low_res="10m",
        high_res="05m",
        output_base_dir=output_path,
    )

    # Mostrar resumen de cantidad de muestras por división
    print("counts")

    # s2v.generate_dataset(
    #     dir_sen2venus_path=s2v_filtered_path,
    #     sites=['FGMANAUS'],
    #     dir_OutputData_path=output_path,  # paso la ruta absoluta
    #     output_name='dataset_test2'
    # )


