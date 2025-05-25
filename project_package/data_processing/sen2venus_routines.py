import os
from pathlib import Path
import torch # type: ignore
import random
import gc
import csv
import matplotlib.pyplot as plt # type: ignore
import pandas as pd
import numpy as np
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset # type: ignore
import torchvision.transforms.functional as functional_transforms # type: ignore
from tqdm import tqdm # type: ignore



import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

def extract_data_Sen2Venus(dir_sen2venus_path, dir_OutputData_path):
    '''
    Processes the Sen2Venus dataset to prepare data for a super-resolution task using only bands B2, B3, and B4 (RGB).

    For each site, the function generates and stores tensors in separate directories corresponding to:
      - High-resolution (5m) images from Venus
      - Medium-resolution (10m) images from Sentinel-2
      - Low-resolution (20m) images obtained by downscaling Sentinel-2 data

    Parameters:
    dir_sen2venus_path (str): Path to the directory containing the raw Sen2Venus dataset.
    dir_OutputData_path (str): Path where the processed dataset will be saved.

    Returns:
    samples_number (int): Total number of samples extracted (based on one resolution).
    total_size (float): Total size of all processed samples in gigabytes (GB).
    '''

    sites = [nombre for nombre in os.listdir(dir_sen2venus_path) if os.path.isdir(os.path.join(dir_sen2venus_path, nombre))]

    os.makedirs(dir_OutputData_path, exist_ok=True)

    total_samples = 0
    total_size_bytes = 0

    for site in sites:
        site_input_path = os.path.join(dir_sen2venus_path, site)

        csv_files = [f for f in os.listdir(site_input_path) if f.endswith('.csv')]
        if len(csv_files) != 1:
            print(f"⚠️ Error en '{site_input_path}': se esperaba 1 CSV, se encontraron {len(csv_files)}")
            continue

        df = pd.read_csv(os.path.join(site_input_path, csv_files[0]), sep='\t')

        site_output_path = os.path.join(dir_OutputData_path, site)
        os.makedirs(site_output_path, exist_ok=True)

        site_output_path_5m = os.path.join(site_output_path, '5m')
        os.makedirs(site_output_path_5m, exist_ok=True)

        site_output_path_10m = os.path.join(site_output_path, '10m')
        os.makedirs(site_output_path_10m, exist_ok=True)

        site_output_path_20m = os.path.join(site_output_path, '20m')
        os.makedirs(site_output_path_20m, exist_ok=True)

        # Drop unwanted bands columns and rename columns to keep only B2,B3,B4 bands
        df = df.drop(['tensor_05m_b5b6b7b8a', 'tensor_20m_b5b6b7b8a'], axis=1, errors='ignore')
        df.rename(columns={'tensor_05m_b2b3b4b8': 'tensor_05m_b2b3b4'}, inplace=True)
        df.rename(columns={'tensor_10m_b2b3b4b8': 'tensor_10m_b2b3b4'}, inplace=True)

        for index, row in tqdm(df.iterrows(), desc=f"Procesing tensors for site {site}", total=len(df)):
            # 5m patches (Venus)
            filename_5m = row['tensor_05m_b2b3b4']
            input_file_path_5m = os.path.join(site_input_path, filename_5m)
            try:
                tensor_5m = torch.load(input_file_path_5m)
                tensor_5m = tensor_5m[:, [2, 1, 0], :, :]  # Convert BGR to RGB

                name_wo_ext = filename_5m[:-3]
                new_name_5m = name_wo_ext[:-2] + '.pt'
                output_file_path_5m = os.path.join(site_output_path_5m, new_name_5m)

                df.at[index, 'tensor_05m_b2b3b4'] = os.path.join('5m', new_name_5m)

                torch.save(tensor_5m, output_file_path_5m)
                total_size_bytes += os.path.getsize(output_file_path_5m)
            except Exception as e:
                print(f"Error procesando {input_file_path_5m}: {e}")

            # 10m patches (Sentinel-2)
            filename_10m = row['tensor_10m_b2b3b4']
            input_file_path_10m = os.path.join(site_input_path, filename_10m)
            try:
                tensor_10m = torch.load(input_file_path_10m)
                tensor_10m = tensor_10m[:, [2, 1, 0], :, :]  # Convert BGR to RGB

                name_wo_ext = filename_10m[:-3]
                new_name_10m = name_wo_ext[:-2] + '.pt'
                output_file_path_10m = os.path.join(site_output_path_10m, new_name_10m)

                df.at[index, 'tensor_10m_b2b3b4'] = os.path.join('10m', new_name_10m)

                torch.save(tensor_10m, output_file_path_10m)
                total_size_bytes += os.path.getsize(output_file_path_10m)
            except Exception as e:
                print(f"Error procesando {input_file_path_10m}: {e}")

            # 20m patches (downsampled from 10m)
            try:
                downsampled_tensor = F.interpolate(tensor_10m.float(), scale_factor=0.5, mode='bilinear', align_corners=False)
                downsampled_tensor = downsampled_tensor.short()

                name_wo_ext = filename_10m[:-3]
                new_name_20m = name_wo_ext.replace('10m', '20m')[:-2] + '.pt'
                output_file_path_20m = os.path.join(site_output_path_20m, new_name_20m)

                df.at[index, 'tensor_20m_b2b3b4'] = os.path.join('20m', new_name_20m)

                torch.save(downsampled_tensor, output_file_path_20m)
                total_size_bytes += os.path.getsize(output_file_path_20m)
            except Exception as e:
                print(f"Error processing 20m patch derived from {input_file_path_10m}: {e}")

        csv_path = os.path.join(site_output_path, site + '.csv')
        df.to_csv(csv_path, index=False)

        # Sumar el número de muestras del sitio (usamos la cantidad de filas del DataFrame)
        # total_samples += len(df)
        total_samples += df['nb_patches'].sum()

    total_size_gb = total_size_bytes / (1024 ** 3)  # Bytes a GB

    return total_samples, total_size_gb



def generate_dataset(dir_sen2venus, sites, output_folder, output_name='my_dataset'):
    '''
    Combines multi-image tensors from multiple sites into one dataset per resolution (5m and 10m),
    and stores them with metadata in a specified output directory.

    Parameters:
    - dir_sen2venus (str): Path to the sen2vnus directory containing processed site folders.
    - sites (list of str): List of site folder names to include.
    - output_folder (str): Directory where the output dataset folder will be created.
    - output_name (str): Name of the dataset subfolder inside the output_folder.

    Returns:
    - str: Path to the created dataset folder containing combined tensors and metadata.
    '''

    resolutions = ['05m', '10m', '20m']
    all_tensors = {'05m': [], '10m': [], '20m': []}
    metadata = []

    # Create output folder and subfolder
    os.makedirs(output_folder, exist_ok=True)
    dataset_dir = os.path.join(output_folder, output_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for site in sites:
        csv_path = os.path.join(dir_sen2venus, site, f'{site}.csv')
        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing CSV for site '{site}', skipping.")
            continue

        df = pd.read_csv(csv_path)

        for res in resolutions:
            column_name = f'tensor_{res}_b2b3b4'
            if column_name not in df.columns:
                print(f"[WARNING] Column '{column_name}' missing in CSV for site '{site}', skipping.")
                continue

            for rel_path in tqdm(df[column_name], desc=f"{site} ({res})"):
                abs_path = os.path.join(dir_sen2venus, site, rel_path)
                try:
                    tensor = torch.load(abs_path)  # Expected shape: [N, 3, H, W]
                    if tensor.dim() != 4 or tensor.shape[1] != 3:
                        print(f"[WARNING] Invalid shape {tensor.shape} in {abs_path}, skipping.")
                        continue

                    num_images = tensor.shape[0]
                    all_tensors[res].append(tensor)

                    metadata.append({
                        'site': site,
                        'tensor_path': abs_path,
                        'num_images': num_images,
                        'resolution': res
                    })

                except Exception as e:
                    print(f"[ERROR] Failed to load {abs_path}: {e}")


    # Save combined tensors
    for res in resolutions:
        if not all_tensors[res]:
            print(f"[WARNING] No tensors loaded for resolution {res}.")
            continue

        combined_tensor = torch.cat(all_tensors[res], dim=0)
        tensor_output_path = os.path.join(dataset_dir, f'{res}.pt')
        torch.save(combined_tensor, tensor_output_path)
        print(f"[INFO] Saved combined tensor for {res} to: {tensor_output_path}")

    # Save metadata CSV
    csv_path = os.path.join(dataset_dir, f'{output_name}.csv')
    pd.DataFrame(metadata).to_csv(csv_path, index=False)
    print(f"[INFO] Saved metadata CSV to: {csv_path}")

    # Verify sample count consistency between resolutions
    count_5m = sum(t.shape[0] for t in all_tensors['05m'])
    count_10m = sum(t.shape[0] for t in all_tensors['10m'])
    count_20m = sum(t.shape[0] for t in all_tensors['20m'])

    del all_tensors

    if count_5m != count_10m or count_5m != count_20m:
        print(f"[WARNING] Mismatch in sample count: 5m={count_5m}, 10m={count_10m}, 20m={count_20m}")
    else:
        print(f"[INFO] Total samples: {count_5m} (matching in 5m, 10m, and 20m)")

    return dataset_dir, count_5m



def load_files_tensor_data(dataset_dir, low_res_file, high_res_file, scale_value=10000):
    """
    Loads low-resolution and high-resolution image tensors from disk, applies normalization, 
    resizes the low-resolution tensors using bicubic interpolation, and returns the processed tensors.

    Parameters:
    -----------
    dataset_dir : str or Path
        Directory path where the `.pt` tensor files are located.

    low_res_file : str
        Filename (without extension) of the low-resolution tensor file to load.
        The tensor is expected to have shape (N, 4, 128, 128) where N is the number of images 
        and 4 is the number of spectral bands.

    high_res_file : str
        Filename (without extension) of the high-resolution tensor file to load.
        The tensor is expected to have shape (N, 4, 256, 256).

    scale_value : float, optional
        Value to scale down the tensor values (default is 10000), commonly used to normalize 
        reflectance or scientific measurements.

    Returns:
    --------
    resized_tensor_low_res : torch.Tensor
        Low-resolution tensor resized to (256, 256) per image using bicubic interpolation,
        normalized to the range [0, 1]. Shape: (N, 3, 256, 256)

    tensor_high_res : torch.Tensor
        High-resolution tensor normalized to the range [0, 1]. Shape: (N, 3, 256, 256)

    Notes:
    ------
    - Both tensors are min-max normalized per image and per channel.
    - After resizing, the low-resolution tensor is clamped to [0, 1] to remove possible artifacts
      introduced by bicubic interpolation.
    """

    data_folder = Path(dataset_dir)

    tensor_low_res = torch.load(os.path.join(data_folder,low_res_file,'.pt'))
    tensor_low_res = torch.load(os.path.join(data_folder,high_res_file,'.pt'))
    
    tensor_low_res=tensor_low_res/scale_value
    tensor_high_res=tensor_high_res/scale_value
    
    max_val_low_res = tensor_low_res.amax(dim=(2, 3))  
    min_val_low_res = tensor_low_res.amin(dim=(2, 3))   
    
    max_val_high_res = tensor_high_res.amax(dim=(2, 3))  
    min_val_high_res = tensor_high_res.amin(dim=(2, 3)) 
    
    tensor_low_res = (tensor_low_res - min_val_low_res[:, :, None, None]) / (max_val_low_res[:, :, None, None] - min_val_low_res[:, :, None, None])
    tensor_high_res = (tensor_high_res - min_val_high_res[:, :, None, None]) / (max_val_high_res[:, :, None, None] - min_val_high_res[:, :, None, None])
    
    del max_val_low_res
    del min_val_low_res
    del max_val_high_res
    del min_val_high_res
    
    #Interpolate low resolution tensors (128x128) to (256x256) using bicubic interpolation
    resized_tensor_low_res = torch.stack([functional_transforms.resize(img, (256, 256), \
                                         interpolation=functional_transforms.InterpolationMode.BICUBIC) for img in tensor_low_res])
    
    del tensor_low_res
    
    resized_tensor_low_res.clamp(0,1)
    tensor_high_res.clamp(0,1)

    return resized_tensor_low_res,tensor_high_res