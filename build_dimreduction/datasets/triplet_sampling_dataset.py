import logging.config
import os.path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from ete4 import Tree
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from evaluation_visualization.analysis_pipeline import _convert_to_full
from evaluation_visualization.tree_building import TreeBuilder
from inference_pipeline.full_pipeline import get_input_data

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def _sort_and_group_pairings(pairs, distances, desc=False):
    # Calculate the indices that would sort the distances array
    def implode_with_np(pairs_arr: np.array):
        """Implode a np array using numpy
        [0 ,1]
        [0, 2]
        Becomes:
        0:[0,1]
        """
        first_col = pairs_arr[:, 0]
        # get a list of all unique values
        unique_values = np.unique(first_col)
        # get indices where values change
        indices = np.searchsorted(first_col, unique_values)
        # cut df where the first value changes
        index_lists = np.split(pairs_arr[:, 1], indices[1:])

        return {anchor: pairables for anchor, pairables in zip(unique_values, index_lists)}

    # create dict with pairing partners
    pairs = np.stack(pairs).T
    pairs_dict = implode_with_np(pairs)

    # sort the values from smallest to larges
    # sort all distances along columns
    if desc:
        distances = -distances
    ordered_dists = distances.argsort(axis=1)

    # update order for each entry
    for x in pairs_dict.keys():
        # get indices where the index from the distmatrix is in the pairing index list
        # only keep these indices from the order dist indices
        ordered_indices = ordered_dists[x][np.where(np.isin(ordered_dists[x], pairs_dict[x]))]
        pairs_dict[x] = ordered_indices

    return pairs_dict


class TripletSamplingDataset(Dataset):

    def __init__(self, model: Literal['prott5', 'esm'], path_to_input_data: str, device):
        path_to_fasta, path_to_gt_distances = get_input_data(path_to_input_data)
        assert os.path.isfile(path_to_fasta), "Path to fasta does not exist"
        assert os.path.isfile(path_to_gt_distances), "Path to distances does not exist"
        embedder = ProtSeqEmbedder(model)
        embedd_data = zip(*embedder.get_raw_embeddings(path_to_fasta))
        self.ids, self.prott5_embeddings = list(embedd_data)
        self.device = device

        self.cophentic_distances = self._get_cophentic_distmatrix(path_to_gt_distances)
        self.positive_threshold = None
        self.negative_threshold = None

        self.positive_gt_pairs = None
        self.negative_gt_pairs = None

        self.positive_embedd_pairs = None
        self.negative_embedd_pairs = None

    def _compute_pos_neg_pairs(self, distance_matrix: np.array, positive_condition: np.array, negative_condition: np.array, desc=False):

        postive_pairs = np.where(positive_condition)
        negative_pairs = np.where(negative_condition)

        pos_pairings = _sort_and_group_pairings(postive_pairs, distance_matrix, desc=desc)
        neg_pairings = _sort_and_group_pairings(negative_pairs, distance_matrix, desc=desc)

        return pos_pairings, neg_pairings

    def set_gt_pairings(self):
        # Order gt pairings in decending order
        # Positives: Most similar has smallest index
        # Negatives: Most similar has smallest index
        positive_condition = ((self.cophentic_distances < self.positive_threshold) & (self.cophentic_distances > 0))
        negative_condition = self.cophentic_distances > self.negative_threshold

        self.positive_gt_pairs, self.negative_gt_pairs = self._compute_pos_neg_pairs(distance_matrix=self.cophentic_distances,
                                                                                     positive_condition=positive_condition,
                                                                                     negative_condition=negative_condition,
                                                                                     desc=True)

    def set_embedding_pairings(self, model_forward) -> dict:
        """
        This method computes positive and negative pairing embeddings for all pairings.
        1. First all embeddings are generated
        2. The cosine distance [0,2] is computed across all embeddings
        3. The thresholds for want counts as pos and neg are computed
        4. The distances and thresholds are passed of to pair finding
        Args:
            model_forward: A forward funtion of a model

        Returns:
            The sampling thresholds
        """
        phy_embedds = []
        for prott5_embedd in self.prott5_embeddings:
            phy_embedds.append(model_forward(prott5_embedd.to(self.device)))

        norm_embeddings = F.normalize(torch.stack(phy_embedds), p=2, dim=1)
        embedding_space_dist = 1 - torch.mm(norm_embeddings, norm_embeddings.t()).data.cpu().numpy()

        # define postive theshold dynamically
        flat_dists = np.sort(embedding_space_dist.flatten())
        # get the index of the 4000 smallest value
        pos_threshold_index = 4000
        zero_index = min(np.abs(flat_dists-1).argmin(),len(flat_dists)-2001)
        neg_lower_index = zero_index - 2000
        neg_upper_index = zero_index + 2000

        pos_threshold = flat_dists[pos_threshold_index]
        neg_lower_threshold = max(0.8,flat_dists[neg_lower_index])
        neg_upper_threshold = min(1.2,flat_dists[neg_upper_index])


        # define conditions on positive and negative pairs
        positive_condition = embedding_space_dist > pos_threshold
        negative_condition = ((embedding_space_dist < neg_upper_threshold) & (embedding_space_dist > neg_lower_threshold))

        # Order gt pairings in ascending order
        # Positives: Most similar has largest index
        # Negatives: Most similar has largest index
        self.positive_embedd_pairs, self.negative_embedd_pairs = self._compute_pos_neg_pairs(distance_matrix=embedding_space_dist,
                                                                                             positive_condition=positive_condition,
                                                                                             negative_condition=negative_condition)

        return {"pos_embedd_threshold":pos_threshold,
                "up_neg_embedd_threshold":neg_upper_threshold,
                "low_neg_embedd_threshold":neg_lower_threshold}

    def set_thresholds(self, pos_threshold: float, neg_threshold: float):
        self.positive_threshold = pos_threshold
        self.negative_threshold = neg_threshold

    def _get_cophentic_distmatrix(self, path_to_distances) -> np.ndarray:
        logger.debug(f"Loading distance matrix from {path_to_distances}")
        treebuilder = TreeBuilder(_convert_to_full(pd.read_csv(path_to_distances, index_col=0)), is_truth=True)
        newick_rep = treebuilder.compute_tree()
        t = Tree(newick_rep.format("newick"), parser=1)
        cophentic_distances, names = t.cophenetic_matrix()

        return pd.DataFrame(cophentic_distances, index=names, columns=names).reindex(self.ids, axis=0).reindex(self.ids, axis=1).values

    def __getitem__(self, idx):
        if idx in self.positive_embedd_pairs.keys() and idx in self.positive_gt_pairs.keys():
            possible_pos_partners = np.stack(
                np.intersect1d(self.positive_gt_pairs[idx], self.positive_embedd_pairs[idx], return_indices=True, assume_unique=True)).T
            if possible_pos_partners.shape[0] > 0:
                pos_diffs = np.abs(possible_pos_partners[:, 1] - possible_pos_partners[:, 2])
                max_diff = np.argmax(pos_diffs)
                positive_index = possible_pos_partners[max_diff, 0]
            else:
                positive_index = idx
        else:
            # return the anchor so there is not difference and therefore no loss
            positive_index = idx

        if idx in self.negative_embedd_pairs.keys() and idx in self.negative_gt_pairs.keys():
            possible_neg_partners = np.stack(
                np.intersect1d(self.negative_gt_pairs[idx], self.negative_embedd_pairs[idx], return_indices=True, assume_unique=True)).T
            if possible_neg_partners.shape[0] > 0:
                neg_diffs = np.abs(possible_neg_partners[:, 1] - possible_neg_partners[:, 2])
                max_diff = np.argmax(neg_diffs)
                negative_index = possible_neg_partners[max_diff, 0]
            else:
                negative_index = np.random.randint(0, self.cophentic_distances.shape[0])
        else:
            negative_index = np.random.randint(0, self.cophentic_distances.shape[0])

        sample = torch.stack(
            [self.prott5_embeddings[idx][None, :],
             self.prott5_embeddings[positive_index][None, :],
             self.prott5_embeddings[negative_index][None, :]], dim=0
        )
        if sample is None:
            print('hi')

        return sample

    def __len__(self):
        return self.cophentic_distances.shape[0]
