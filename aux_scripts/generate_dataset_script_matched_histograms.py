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

from dump import train_common_routines_OLD as tcr
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

    #Patches
    patching = True
    patch_size = {'low':(32,32),
                  'high':(64,64)}
    stride = {'low':(24,24),
              'high':(48,48)}
    

    # ------------------------------------------------
    # Dataset 1: Campo
    #-------------------------------------------------
    dataset_name = 'Dataset_Campo_patched_MatchedHist'
    # Directorio de salida donde se generará el dataset
    output_path = os.path.join(project_dir, 'datasets', dataset_name)
    # output_path = os.path.join(project_dir, 'datasets')
    os.makedirs(output_path, exist_ok=True)  # Crear si no existe

    # Sitios a procesar (podés agregar más si querés)
    selected_sites = ['ARM']


    # Generar dataset en formato .tar usando WebDataset
    counts = s2v.generate_dataset_targenerate_dataset_tar_with_histogram_matching(
        dir_sen2venus_path=s2v_filtered_path,
        sites=selected_sites,
        low_res="10m",
        high_res="05m",
        output_base_dir=output_path,
        max_samples_per_shard=5000,
        interpolation=False,
        patching=patching,
        patch_size=patch_size,
        stride=stride
    )


    # ------------------------------------------------
    # Dataset 2: Desierto   
    #-------------------------------------------------
    dataset_name = 'Dataset_Desierto_patched_MatchedHist'
    # Directorio de salida donde se generará el dataset
    output_path = os.path.join(project_dir, 'datasets', dataset_name)
    # output_path = os.path.join(project_dir, 'datasets')
    os.makedirs(output_path, exist_ok=True)  # Crear si no existe

    # Sitios a procesar (podés agregar más si querés)
    selected_sites = ['BAMBENW2']

    #Patches
    patching = True
    patch_size = {'low':(32,32),
                  'high':(64,64)}
    stride = {'low':(16,16),
              'high':(32,32)}
    
    # Generar dataset en formato .tar usando WebDataset
    counts = s2v.generate_dataset_targenerate_dataset_tar_with_histogram_matching(
        dir_sen2venus_path=s2v_filtered_path,
        sites=selected_sites,
        low_res="10m",
        high_res="05m",
        output_base_dir=output_path,
        max_samples_per_shard=1500,
        interpolation=False,
        patching=patching,
        patch_size=patch_size,
        stride=stride
    )


    # ------------------------------------------------
    # Dataset 3: Montana
    #-------------------------------------------------
    dataset_name = 'Dataset_Montana_patched_MatchedHist'
    # Directorio de salida donde se generará el dataset
    output_path = os.path.join(project_dir, 'datasets', dataset_name)
    # output_path = os.path.join(project_dir, 'datasets')
    os.makedirs(output_path, exist_ok=True)  # Crear si no existe

    # Sitios a procesar (podés agregar más si querés)
    selected_sites = ['ES-LTERA','NARYN','SUDOUE-4','SUDOUE-5','SUDOUE-6']

    
    # Generar dataset en formato .tar usando WebDataset
    counts = s2v.generate_dataset_targenerate_dataset_tar_with_histogram_matching(
        dir_sen2venus_path=s2v_filtered_path,
        sites=selected_sites,
        low_res="10m",
        high_res="05m",
        output_base_dir=output_path,
        max_samples_per_shard=5000,
        interpolation=False,
        patching=patching,
        patch_size=patch_size,
        stride=stride
    )