
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
import pathlib

# ───────────────────────────────────────────────────────────────────────────────
# 🌍 Third-Party Library Imports
# ───────────────────────────────────────────────────────────────────────────────
import torch                # PyTorch: deep learning framework
import geopandas as gpd     # For handling geospatial data with GeoDataFrames
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import webdataset as wds
import tarfile
import glob


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





class PtWebDataset:
    def __init__(self, tar_path_or_pattern, length, batch_size=2, shuffle_buffer=10, shuffle=True):
        """
        Inicializa el pipeline de WebDataset con soporte para múltiples archivos .tar.

        Args:
            tar_path_or_pattern (str o list): Ruta(s) a archivo(s) tar o patrón glob (ejemplo: 'train-*.tar').
            length (int): Número total de muestras.
            batch_size (int): Tamaño del batch.
            shuffle_buffer (int): Tamaño del buffer para shuffle.
            shuffle (bool): Habilitar/deshabilitar shuffle.
        """
        if isinstance(tar_path_or_pattern, str):
            if '*' in tar_path_or_pattern or '?' in tar_path_or_pattern or '[' in tar_path_or_pattern:
                self.tar_paths = sorted(glob.glob(tar_path_or_pattern))
            else:
                self.tar_paths = [tar_path_or_pattern]
        elif isinstance(tar_path_or_pattern, list):
            self.tar_paths = tar_path_or_pattern
        else:
            raise ValueError("tar_path_or_pattern debe ser string o lista de rutas.")

        # Convertir rutas a absolutas, barras '/' y agregar prefijo 'file://'
        self.tar_paths = [
            "file://" + str(pathlib.Path(p).absolute()).replace("\\", "/")
            for p in self.tar_paths
        ]

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle = shuffle
        self.length = length

        self.dataset = self._create_dataset()



    @staticmethod
    def _decode_pt(byte_data):
        """
        Decodes raw bytes into a PyTorch tensor and converts it to float.

        Args:
            byte_data (bytes): Raw byte content of a tensor saved with torch.save.

        Returns:
            torch.Tensor: Decoded tensor converted to float.
        """
        return torch.load(io.BytesIO(byte_data)).float()

    def _create_dataset(self):
        """
        Creates the WebDataset pipeline with decoding and batching.

        Applies:
            - Loads dataset from tar_path.
            - Converts samples to tuples (image_pt, label_pt).
            - Decodes bytes to float tensors.
            - Applies shuffle if enabled.
            - Applies batching.

        Returns:
            wds.DataPipeline: A WebDataset pipeline ready for DataLoader consumption.
        """
        pipeline = (
            wds.WebDataset(self.tar_paths)
            .to_tuple("pt_input.pt", "pt_output.pt")
            .map_tuple(self._decode_pt, self._decode_pt)
            .with_length(self.length)
        )
        if self.shuffle:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        pipeline = pipeline.batched(self.batch_size)
        return pipeline

    def get_dataloader(self, num_workers=2):
        """
        Returns a WebLoader that iterates over the WebDataset pipeline.

        Args:
            num_workers (int): Number of worker processes for data loading.

        Returns:
            webdataset.WebLoader: A WebLoader ready for training or validation.
        """
        return wds.WebLoader(self.dataset, num_workers=num_workers, batch_size=None)

    def __iter__(self):
        """
        Allows direct iteration over the PtWebDataset instance.

        Example:
            >>> dataset = PtWebDataset("data.tar")
            >>> for images, labels in dataset:
            >>>     # Training code here

        Returns:
            iterator: An iterator over batches from the DataLoader.
        """
        return iter(self.get_dataloader())
