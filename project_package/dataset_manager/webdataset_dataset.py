
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
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import webdataset as wds
import tarfile


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




class PtWebDataset:
    """
    WebDataset-based dataset for loading tensors saved as `.pt` files inside a tar archive.

    Each sample is expected to have two files: a tensor image `.pt` and a tensor label `.pt`.
    Tensors are loaded from bytes using torch.load with io.BytesIO and converted to float.

    Parameters:
    -----------
    tar_path : str
        Path to the tar file containing the dataset.
    batch_size : int, optional (default=2)
        Batch size for the DataLoader.
    shuffle_buffer : int, optional (default=10)
        Buffer size used for shuffling the samples.
    shuffle : bool, optional (default=True)
        Whether to shuffle the dataset.

    Example:
    --------
    >>> dataset = PtWebDataset("data.tar", batch_size=4)
    >>> dataloader = dataset.get_dataloader()
    >>> for images, labels in dataloader:
    >>>     print(images.shape)
    """

    def __init__(self, tar_path, length, batch_size=2, shuffle_buffer=10, shuffle=True):
        """
        Initializes the WebDataset pipeline with the given parameters.

        Args:
            tar_path (str): Path to the tar file containing the dataset.
            batch_size (int): Batch size.
            shuffle_buffer (int): Buffer size for shuffling.
            shuffle (bool): Flag to enable shuffling.
        """
        self.tar_path = tar_path
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
            wds.WebDataset(self.tar_path)
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
