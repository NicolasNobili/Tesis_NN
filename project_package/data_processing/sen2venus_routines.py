# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import os
import io
import json
import tarfile
import random

# ───────────────────────────────────────────────────────────────────────────────
# 📚 Data & Visualization
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 PyTorch & Deep Learning
# ───────────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────────
# 🧪 Sen2Venus Processing Routines
# ───────────────────────────────────────────────────────────────────────────────


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
        site_output_path_5m = os.path.join(site_output_path, '5m')
        site_output_path_10m = os.path.join(site_output_path, '10m')
        site_output_path_20m = os.path.join(site_output_path, '20m')
        for p in [site_output_path, site_output_path_5m, site_output_path_10m, site_output_path_20m]:
            os.makedirs(p, exist_ok=True)

        # Drop unwanted bands columns and rename columns to keep only B2,B3,B4 bands
        df.drop(columns=['tensor_05m_b5b6b7b8a', 'tensor_20m_b5b6b7b8a'], errors='ignore', inplace=True)
        df.rename(columns={
            'tensor_05m_b2b3b4b8': 'tensor_05m_b2b3b4',
            'tensor_10m_b2b3b4b8': 'tensor_10m_b2b3b4'
        }, inplace=True)

        for index, row in tqdm(df.iterrows(), desc=f"Procesing tensors for site {site}", total=len(df)):
            # 5m patches (Venus)
            filename_5m = row['tensor_05m_b2b3b4']
            input_file_path_5m = os.path.join(site_input_path, filename_5m)
            try:
                tensor_5m = torch.load(input_file_path_5m)[:, [2, 1, 0], :, :]  # Convert BGR to RGB 
                new_name_5m = filename_5m[:-5] + '.pt'
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
                tensor_10m = torch.load(input_file_path_10m) [:, [2, 1, 0], :, :]  # Convert BGR to RG
                new_name_10m = filename_10m[:-5] + '.pt'
                output_file_path_10m = os.path.join(site_output_path_10m, new_name_10m)
                df.at[index, 'tensor_10m_b2b3b4'] = os.path.join('10m', new_name_10m)
                torch.save(tensor_10m, output_file_path_10m)
                total_size_bytes += os.path.getsize(output_file_path_10m)
            except Exception as e:
                print(f"Error procesando {input_file_path_10m}: {e}")

            # 20m patches (downsampled from 10m)
            try:
                downsampled_tensor = F.interpolate(tensor_10m.float(), scale_factor=0.5, mode='bilinear', align_corners=False).short()
                new_name_20m = filename_10m.replace('10m', '20m')[:-5] + '.pt'
                output_file_path_20m = os.path.join(site_output_path_20m, new_name_20m)
                df.at[index, 'tensor_20m_b2b3b4'] = os.path.join('20m', new_name_20m)
                torch.save(downsampled_tensor, output_file_path_20m)
                total_size_bytes += os.path.getsize(output_file_path_20m)
            except Exception as e:
                print(f"Error processing 20m patch derived from {input_file_path_10m}: {e}")

        df.to_csv(os.path.join(site_output_path, site + '.csv'), index=False)
        total_samples += df['nb_patches'].sum()

    total_size_gb = total_size_bytes / (1024 ** 3)  # Bytes a GB

    return total_samples, total_size_gb




def generate_dataset(dir_sen2venus_path, sites, dir_OutputData_path, output_name='my_dataset'):
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
    os.makedirs(dir_OutputData_path, exist_ok=True)
    dataset_dir = os.path.join(dir_OutputData_path, output_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for site in sites:
        csv_path = os.path.join(dir_sen2venus_path, site, f'{site}.csv')
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
                abs_path = os.path.join(dir_sen2venus_path, site, rel_path)
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




def load_files_tensor_data( low_res_file, high_res_file, scale_value=10000):
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

    tensor_low_res = torch.load(low_res_file, weights_only=True)
    tensor_high_res = torch.load(high_res_file, weights_only=True)
    
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
    resized_tensor_low_res = F.interpolate(tensor_low_res , size=(256,256), mode='bicubic',align_corners=False)
    
    del tensor_low_res
    
    resized_tensor_low_res.clamp(0,1)
    tensor_high_res.clamp(0,1)

    return resized_tensor_low_res,tensor_high_res




# ───────────────────────────────────────────────────────────────────────────────
# 📦 WebDataset-related Functions
# These functions are designed to generate and manipulate WebDataset tar archives.
# ───────────────────────────────────────────────────────────────────────────────


def generate_dataset_tar(
        dir_sen2venus_path,
        sites, low_res,
        high_res,
        tar_output_path
):
    """
    Generates a WebDataset tarball containing paired tensor data from specified sites.

    For each site, reads CSV metadata pointing to tensor files saved as PyTorch `.pt` files.
    Extracts paired tensors for low and high resolution images, slices them into individual
    samples, and stores them as serialized `.pt` files inside a tar archive.

    Args:
        dir_sen2venus_path (str): Root directory path containing site folders and CSV metadata.
        sites (list of str): List of site names to process.
        low_res (str): Resolution label for the input tensors (e.g., "10m").
        high_res (str): Resolution label for the output tensors (e.g., "05m").
        tar_output_path (str): File path where the output tarball will be saved.

    Returns:
        tuple: (tar_output_path, int)
            - tar_output_path (str): Path to the generated tar archive.
            - int: Number of image-label pairs added to the tarball.

    Workflow:
        - Iterates through each site folder.
        - Loads site CSV metadata to locate paired tensor file paths.
        - Loads low and high resolution tensor files using torch.load().
        - Checks if the number of samples matches between low and high resolution tensors.
        - Iterates over each sample in the tensor batches.
        - Serializes individual tensor samples as `.pt` files into an in-memory buffer.
        - Adds serialized tensors to the tarball with structured paths (input/output folders).
        - Keeps a global count of processed samples.
        - Handles missing files, mismatched shapes, or errors gracefully by skipping and logging.

    Notes:
        - Tensors are saved as `short` type to reduce storage size.
        - The function uses in-memory buffers (BytesIO) for efficient tar file creation without temp files.
    """
    counts = 0  # Global sample counter

    with tarfile.open(tar_output_path, "w") as tar:
        for site in sites:
            csv_path = os.path.join(dir_sen2venus_path, site, f"{site}.csv")
            if not os.path.exists(csv_path):
                print(f"[WARNING] Missing CSV for site '{site}', skipping.")
                continue

            df = pd.read_csv(csv_path)

            col_low = f'tensor_{low_res}_b2b3b4'
            col_high = f'tensor_{high_res}_b2b3b4'

            if col_low not in df.columns or col_high not in df.columns:
                print(f"[WARNING] Missing required columns ({col_low}, {col_high}) in CSV for site '{site}', skipping.")
                continue

            for path_low, path_high in tqdm(zip(df[col_low], df[col_high]), total=len(df), desc=f"Processing {site}"):
                abs_path_10m = os.path.join(dir_sen2venus_path, site, path_low)
                abs_path_05m = os.path.join(dir_sen2venus_path, site, path_high)

                try:
                    tensor_10m = torch.load(abs_path_10m)  # [N, 3, H, W]
                    tensor_05m = torch.load(abs_path_05m)

                    if tensor_10m.shape[0] != tensor_05m.shape[0]:
                        print(f"[WARNING] Sample count mismatch in {site}, skipping.")
                        continue

                    for i in range(tensor_10m.shape[0]):
                        # Serialize input tensor sample as short to reduce size
                        img_10m = tensor_10m[i].short()
                        img_buffer = io.BytesIO()
                        torch.save(img_10m, img_buffer)
                        img_buffer.seek(0)

                        img_tarinfo = tarfile.TarInfo(name=f"{counts:08d}.pt_input.pt")
                        img_tarinfo.size = img_buffer.getbuffer().nbytes
                        tar.addfile(img_tarinfo, img_buffer)

                        # Serialize label tensor sample as short
                        label_05m = tensor_05m[i].short()
                        label_buffer = io.BytesIO()
                        torch.save(label_05m, label_buffer)
                        label_buffer.seek(0)

                        label_tarinfo = tarfile.TarInfo(name=f"{counts:08d}.pt_output.pt")
                        label_tarinfo.size = label_buffer.getbuffer().nbytes
                        tar.addfile(label_tarinfo, label_buffer)

                        counts += 1

                except Exception as e:
                    print(f"[ERROR] Failed to process {site}: {e}")

    print(f"[INFO] Tarball created with {counts} image-label pairs at {tar_output_path}")

    return tar_output_path, counts




def generate_dataset_tar_split(
    dir_sen2venus_path,
    sites,
    low_res,
    high_res,
    output_base_dir,
    split_ratios=[0.95, 0.04, 0.01],
    scale_value=10000
):
    """
    Generates three tar files containing paired input/output tensors split into training, validation, and test sets,
    along with a metadata.json file summarizing the dataset.

    Parameters:
    -----------
    dir_sen2venus_path : str
        Base directory containing the site folders and CSV files.
    sites : list of str
        List of site names to include in the dataset.
    low_res : str
        Suffix used to identify low-resolution tensor columns in the CSV.
    high_res : str
        Suffix used to identify high-resolution tensor columns in the CSV.
    output_base_dir : str
        Output directory where tar files and metadata will be saved.
    split_ratios : list of float, optional
        Ratios for train/val/test splits. Must sum to 1.0. Default is [0.95, 0.04, 0.01].
    scale_value : int or float, optional
        Value by which to scale down tensor values (e.g., 10000 for Sentinel-2 reflectance normalization).

    Returns:
    --------
    tuple:
        - (str, str, str): Paths to train, val, and test tar files.
        - int: Total number of tensor pairs saved.

    Notes:
    ------
    - For each site, the function reads a CSV file that lists paths to low- and high-resolution tensor files.
    - Each pair of tensors is normalized, resized (low-res only), and saved into one of the three tar archives.
    - A metadata.json file is generated with sample counts and dataset details.
    """

    os.makedirs(output_base_dir, exist_ok=True)

    train_tar_path = os.path.join(output_base_dir, "train.tar")
    val_tar_path = os.path.join(output_base_dir, "val.tar")
    test_tar_path = os.path.join(output_base_dir, "test.tar")

    tar_train = tarfile.open(train_tar_path, "w")
    tar_val = tarfile.open(val_tar_path, "w")
    tar_test = tarfile.open(test_tar_path, "w")

    counts = {"train": 0, "val": 0, "test": 0}
    total_count = 0

    for site in sites:
        csv_path = os.path.join(dir_sen2venus_path, site, f"{site}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] Missing CSV for site: {site}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        col_low = f'tensor_{low_res}_b2b3b4'
        col_high = f'tensor_{high_res}_b2b3b4'

        if col_low not in df.columns or col_high not in df.columns:
            print(f"[WARNING] Required columns not found in CSV for site: {site}, skipping.")
            continue

        for path_low, path_high in tqdm(zip(df[col_low], df[col_high]), total=len(df), desc=f"Processing {site}"):
            abs_path_low = os.path.join(dir_sen2venus_path, site, path_low)
            abs_path_high = os.path.join(dir_sen2venus_path, site, path_high)

            try:
                tensor_low = torch.load(abs_path_low)
                tensor_high = torch.load(abs_path_high)

                if tensor_low.shape[0] != tensor_high.shape[0]:
                    print(f"[WARNING] Sample count mismatch in {site}, skipping this file pair.")
                    continue

                for i in range(tensor_low.shape[0]):
                    r = random.random()
                    if r < split_ratios[0]:
                        split = "train"
                        tar = tar_train
                    elif r < split_ratios[0] + split_ratios[1]:
                        split = "val"
                        tar = tar_val
                    else:
                        split = "test"
                        tar = tar_test

                    # Normalize and resize low-resolution tensor
                    input_array = tensor_low[i].numpy() / scale_value
                    min_vals = input_array.min(axis=(1, 2), keepdims=True)
                    max_vals = input_array.max(axis=(1, 2), keepdims=True)
                    input_array = (input_array - min_vals) / (max_vals - min_vals + 1e-8)
                    input_array_hwc = np.transpose(input_array, (1, 2, 0))
                    resized = cv2.resize(input_array_hwc, (256, 256), interpolation=cv2.INTER_CUBIC)
                    input_tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).float()

                    # Serialize input tensor
                    input_buffer = io.BytesIO()
                    torch.save(input_tensor, input_buffer)
                    input_buffer.seek(0)
                    input_info = tarfile.TarInfo(name=f"{counts[split]:08d}.pt_input.pt")
                    input_info.size = input_buffer.getbuffer().nbytes
                    tar.addfile(input_info, input_buffer)

                    # Normalize high-resolution tensor
                    output_array = tensor_high[i].numpy() / scale_value
                    min_vals_out = output_array.min(axis=(1, 2), keepdims=True)
                    max_vals_out = output_array.max(axis=(1, 2), keepdims=True)
                    output_array = (output_array - min_vals_out) / (max_vals_out - min_vals_out + 1e-8)
                    output_tensor = torch.from_numpy(output_array).float()

                    # Serialize output tensor
                    output_buffer = io.BytesIO()
                    torch.save(output_tensor, output_buffer)
                    output_buffer.seek(0)
                    output_info = tarfile.TarInfo(name=f"{counts[split]:08d}.pt_output.pt")
                    output_info.size = output_buffer.getbuffer().nbytes
                    tar.addfile(output_info, output_buffer)

                    counts[split] += 1
                    total_count += 1

            except Exception as e:
                print(f"[ERROR] Failed to process tensors for {site}: {e}")

    # Close tar archives
    tar_train.close()
    tar_val.close()
    tar_test.close()

    # Save metadata as JSON
    metadata = {
        "splits": {
            "train": {
                "file": os.path.basename(train_tar_path),
                "num_samples": counts["train"]
            },
            "val": {
                "file": os.path.basename(val_tar_path),
                "num_samples": counts["val"]
            },
            "test": {
                "file": os.path.basename(test_tar_path),
                "num_samples": counts["test"]
            }
        },
        "total_samples": total_count,
        "input_key": "pt_input.pt",
        "output_key": "pt_output.pt",
        "scale_value": scale_value,
        "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Tensor dataset for Sentinel-2 to Venus super-resolution"
    }

    metadata_path = os.path.join(output_base_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[INFO] Saved {total_count} pairs: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"[INFO] Metadata saved to {metadata_path}")
    print(f"[INFO] Dataset tar files saved in: {output_base_dir}")

    return (train_tar_path, val_tar_path, test_tar_path), total_count