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