import logging.config
import os.path
from glob import glob
from typing import Literal

import numpy as np
import pandas as pd
import torch
from ete4 import Tree
from torch.utils.data import Dataset

from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from evaluation_visualization.tree_building import TreeBuilder
from inference_pipeline.full_pipeline import get_input_data
from evaluation_visualization.analysis_pipeline import _convert_to_full

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)




class SamplingDataset(Dataset):
    def __init__(self, model: Literal['prott5', 'esm'], path_to_input_data: str):
        path_to_fasta, path_to_gt_distances = get_input_data(path_to_input_data)
        assert os.path.isfile(path_to_fasta), "Path to fasta does not exist"
        assert os.path.isfile(path_to_gt_distances), "Path to distances does not exist"
        embedder = ProtSeqEmbedder(model)
        embedd_data = zip(*embedder.get_raw_embeddings(path_to_fasta))

        self.ids, self.embeddings = list(embedd_data)
        self.sample_pairs = self._get_sampling_pairs(path_to_gt_distances)

    def __len__(self):
        return len(self.embeddings)

    def _get_sampling_pairs(self, path_to_distances):
        logger.debug("Computing possible sampling pairs")
        cophentic_matrix = self._get_cophentic_distmatrix(path_to_distances)
        cophentic_matrix = cophentic_matrix.reindex(self.ids,axis=0).reindex(self.ids,axis=1)

        cond1 = cophentic_matrix < 0.5
        cond2 = cophentic_matrix > 0
        indices = np.where(cond1 & cond2)

        coordinates = list(zip(indices[0],indices[1]))
        return coordinates

    def _get_cophentic_distmatrix(self, path_to_distances):
        logger.debug(f"Loading distance matrix from {path_to_distances}")
        treebuilder = TreeBuilder(_convert_to_full(pd.read_csv(path_to_distances,index_col=0)), is_truth=True)
        newick_rep = treebuilder.compute_tree()
        t = Tree(newick_rep.format("newick"), parser=1)
        cophentic_distances, names = t.cophenetic_matrix()

        return pd.DataFrame(cophentic_distances,index=names,columns=names)

    def __getitem__(self, idx):
        # Calculate adjusted index for the second embedding
        index1,index2 = self.sample_pairs[idx]
        return torch.cat([self.embeddings[index1][None, :], self.embeddings[index2][None, :]], dim=0)
