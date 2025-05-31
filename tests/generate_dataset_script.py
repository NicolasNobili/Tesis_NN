# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os                   # For file and directory manipulation
import sys                  # To modify Python path for custom module imports
import csv                  # To handle CSV file reading/writing
import random               # For generating random numbers
import numpy as np          # Numerical operations and array handling
import pandas as pd         # DataFrame handling for structured data
import matplotlib.pyplot as plt  # Plotting and visualization
import io

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch                # PyTorch: deep learning framework
import geopandas as gpd     # For handling geospatial data with GeoDataFrames

# ───────────────────────────────────────────────────────────────────────────────
# 🧩 Custom Project Modules
# ───────────────────────────────────────────────────────────────────────────────

# Add custom project folder to system path to enable local module imports
sys.path.append('C:/Users/nnobi/Desktop/FIUBA/Tesis/Project')

# Import common training routines 
from project_package.utils import train_common_routines as tcr

# Import Sentinel-2 to Venus preprocessing utilities
from project_package.data_processing import sen2venus_routines as s2v

# Import general utility functions 
from project_package.utils import utils as utils




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))  # project/test
    project_dir = os.path.abspath(os.path.join(script_dir, '..'))  # subir a project

    s2v_filtered_path = 'C:/Users/nnobi/Desktop/FIUBA/Tesis/Sen2Venus_rgb'

    # output_path = os.path.join(project_dir, 'DATASETS')  # ruta absoluta dentro de project

    # s2v.generate_dataset(
    #     dir_sen2venus_path=s2v_filtered_path,
    #     sites=['FGMANAUS'],
    #     dir_OutputData_path=output_path,  # paso la ruta absoluta
    #     output_name='my_dataset4'
    # )

    output_path = os.path.join(project_dir,'datasets','messi')  # ruta absoluta dentro de project

    # s2v.generate_dataset_tar(
    #     dir_sen2venus_path=s2v_filtered_path,
    #     sites=['FGMANAUS'],
    #     tar_output_path=output_path,
    #     low_res='10m',     # Input
    #     high_res='05m'     # Target
    # )

    counts = s2v.generate_dataset_tar_split(
        dir_sen2venus_path=s2v_filtered_path,
         sites=['FGMANAUS'],
        low_res="10m",
        high_res="05m",
        output_base_dir=output_path,
    )
    print(counts)



