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
    dir_sen2venus_path = 'C:/Users/nnobi/Desktop/FIUBA/Tesis/Sen2Venus_OG'
    sites = [nombre for nombre in os.listdir(dir_sen2venus_path) if os.path.isdir(os.path.join(dir_sen2venus_path, nombre))]

    # Diccionario para guardar coordenadas centroides de cada GPKG por sitio
    site_gpkg_coords = {}

    for site in sites:
        folder_path = os.path.join(dir_sen2venus_path, site)
        gpkg_files = [f for f in os.listdir(folder_path) if f.endswith('.gpkg')]
        
        site_gpkg_coords[site] = {}
        
        for gpkg_file in gpkg_files:
            gpkg_path = os.path.join(folder_path, gpkg_file)
            try:
                gdf = gpd.read_file(gpkg_path)
                valid_geometries = gdf[gdf.geometry.notnull()]
                if not valid_geometries.empty:
                    # Unir geometrías y calcular centroid
                    centroid = valid_geometries.geometry.union_all().centroid

                    # Crear GeoDataFrame para reproyectar el centroide
                    centroid_gdf = gpd.GeoDataFrame(geometry=[centroid], crs=gdf.crs)
                    centroid_wgs84 = centroid_gdf.to_crs(epsg=4326).geometry.iloc[0]

                    latlon = (centroid_wgs84.y, centroid_wgs84.x)  # (latitud, longitud en EPSG:4326)
                    site_gpkg_coords[site][gpkg_file] = latlon
                else:
                    print(f"[WARNING] No valid geometries in: {gpkg_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read GPKG in {gpkg_path}: {e}")

    # Calcular coordenada promedio (lat, lon) por sitio
    site_mean_coords = {}

    for site, gpkg_info in site_gpkg_coords.items():
        coords = list(gpkg_info.values())
        
        if coords:  # Si hay coordenadas disponibles
            mean_lat = sum(coord[0] for coord in coords) / len(coords)
            mean_lon = sum(coord[1] for coord in coords) / len(coords)
            site_mean_coords[site] = (mean_lat, mean_lon)
        else:
            print(f"[WARNING] No valid coordinates for site {site}")

    # Preparar lista para CSV con link directo a Google Maps
    mean_coords_data = []

    for site, (lat, lon) in site_mean_coords.items():
        mean_coords_data.append({
            'site': site,
            'latitude': lat,
            'longitude': lon,
            'google_maps_url': f'https://www.google.com/maps?q={lat},{lon}'
        })

    # Guardar CSV
    df_mean_coords = pd.DataFrame(mean_coords_data)
    df_mean_coords.to_csv('site_mean_coordinates.csv', index=False)

    print("📍 CSV generado: site_mean_coordinates.csv")
