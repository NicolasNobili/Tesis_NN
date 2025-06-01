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
from project_package.utils import train_common_routines2 as tcr

# Import Sentinel-2 to Venus preprocessing utilities
from project_package.data_processing import sen2venus_routines as s2v

# Import general utility functions 
from project_package.utils import utils as utils


if __name__ == "__main__":
    # Process sen2venus dataset to extract only b2,b3 and b4 (rgb images).
    s2v_path = 'C:/Users/nnobi/Desktop/FIUBA/Tesis/Sen2Venus_OG'
    s2v_filtered_path = 'C:/Users/nnobi/Desktop/FIUBA/Tesis/Sen2Venus_rgb'
    s2v.extract_data_Sen2Venus(s2v_path, s2v_filtered_path)
