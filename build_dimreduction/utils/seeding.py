import os
import random
from glob import glob

import numpy as np
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_input_data(path_to_input_folder: str) -> tuple[str, str]:
    assert os.path.isdir(path_to_input_folder), 'Path does not exist!'

    files = glob(os.path.join(path_to_input_folder, '*'))
    fasta_file = [x for x in files if x.endswith('.fasta')][0]
    distance_file = [x for x in files if x.endswith('.csv')][0]

    return fasta_file, distance_file


def zscore_normalize(data):
    """
    Perform Z-score normalization followed by min-max scaling to [0, 1] range on a NumPy array.

    Parameters:
    - data: NumPy array, the input data to be normalized.

    Returns:
    - normalized_data: NumPy array, the Z-score normalized and min-max scaled data.
    """
    # Compute mean and standard deviation for Z-score normalization
    mu = np.mean(data)
    sigma = np.std(data)

    # Z-score normalization
    zscore_normalized_data = (data - mu) / sigma

    # Min-max scaling to [0, 1] range
    min_val = np.min(zscore_normalized_data)
    max_val = np.max(zscore_normalized_data)
    scaled_data = (zscore_normalized_data - min_val) / (max_val - min_val)

    return scaled_data
