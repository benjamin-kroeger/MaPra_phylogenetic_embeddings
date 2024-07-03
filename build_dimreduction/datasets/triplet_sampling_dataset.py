import itertools
import logging.config
import multiprocessing
import os.path
from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ete4 import Tree
from torch.utils.data import Dataset

from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from build_dimreduction.utils.seeding import get_input_data
from evaluation_visualization.analysis_pipeline import _convert_to_full
from evaluation_visualization.tree_building import TreeBuilder

from build_dimreduction.utils.triplet_mining import compute_pos_neg_pairs, embedding_distances, neg_embedd_pairings, pos_embedd_pairings, \
    sampled_triplets, pairing_access_lock

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TripletSamplingDataset(Dataset):

    def __init__(self, model: Literal['prott5', 'esm'], path_to_input_data: str, device):
        # get paths of input files
        path_to_fasta, path_to_gt_distances = get_input_data(path_to_input_data)
        assert os.path.isfile(path_to_fasta), "Path to fasta does not exist"
        assert os.path.isfile(path_to_gt_distances), "Path to distances does not exist"
        # init embedder
        embedder = ProtSeqEmbedder(model)
        # get embedding data
        embedd_data = zip(*embedder.get_raw_embeddings(path_to_fasta))
        self.ids, self.prott5_embeddings = list(embedd_data)
        self.prott5_embeddings = torch.stack(self.prott5_embeddings)
        self.device = device

        # compute ground truth distance matrix using cophentic distance matrix
        self.cophentic_distances = self._get_cophentic_distmatrix(path_to_gt_distances)
        self.embedding_space_distances = np.frombuffer(embedding_distances.get_obj(), dtype=np.float64).reshape(self.cophentic_distances.shape)

        ax = sns.heatmap(self.cophentic_distances)
        ax.set_title('Distance Matrix Truth')
        plt.show()

        self.leeway = None

        self.positive_threshold = None
        self.negative_threshold = None

        self.positive_gt_pairs = None
        self.negative_gt_pairs = None

        self.positive_embedd_pairs = pos_embedd_pairings
        self.negative_embedd_pairs = neg_embedd_pairings

    def set_gt_pairings(self):
        """
        Sets the pairs that will be considered as ground truth postives or neagtives during training
        Returns:
            None, Updates the instance variables

        """
        # Order gt pairings in decending order
        # Positives: Most similar has smallest index
        # Negatives: Most different has smallest index
        positive_condition = ((self.cophentic_distances < self.positive_threshold) & (self.cophentic_distances > 0))
        negative_condition = self.cophentic_distances > self.negative_threshold

        self.positive_gt_pairs, self.negative_gt_pairs = compute_pos_neg_pairs(distance_matrix=self.cophentic_distances,
                                                                               positive_condition=positive_condition,
                                                                               negative_condition=negative_condition,
                                                                               desc=False)

    def set_constants(self, pos_threshold: float, neg_threshold: float, leeway:int):
        """
        Set the cophentic distance threshold for what is considered a positive or a negative pair
        Args:
            pos_threshold: The positive threshold
            neg_threshold: The negative threshold

        Returns:
            None, Updates the instance variables
        """
        self.positive_threshold = pos_threshold
        self.negative_threshold = neg_threshold
        self.leeway = leeway

    def _get_cophentic_distmatrix(self, path_to_distances) -> pd.DataFrame:
        """
        Computes a cophentic distance matrix given a path to distances from an MSA
        1. Computes a Tree based on MSA distances
        2. Calculates all pairwise distances in the tree
        Args:
            path_to_distances: The path to the csv with the distances in a lower triangular format

        Returns:
            A dataframe with sequence names as index and columns and the respective distances
        """
        # build the tree
        logger.debug(f"Loading distance matrix from {path_to_distances}")
        treebuilder = TreeBuilder(_convert_to_full(pd.read_csv(path_to_distances, index_col=0)), is_truth=True)
        newick_rep = treebuilder.compute_tree()
        # read the tree and compute cophentic distances
        t = Tree(newick_rep.format("newick"), parser=1)
        cophentic_distances, names = t.cophenetic_matrix()

        return pd.DataFrame(cophentic_distances, index=names, columns=names).reindex(self.ids, axis=0).reindex(self.ids, axis=1).values

    def __getitem__(self, idx):

        return self.simple_sampling(idx, self.leeway)

    def benjamin_sampling(self, idx, leeway):
        """
        This method returns a triplet sampled from the precomputed pairs
        Args:
            leeway:
            idx:

        Returns:

        """
        # Main idea
        # Given an index the arrays that contain the indices for potential postives are fetched
        # The indices in the GT array are sorted ascendingly, the indices in the EMBEDD array are sorted descendingly
        # The intersection between the 2 arrays is computed with np.1dintersect [intersecting_value,index in arr1, index in arr2]
        # Since the arrays are sorted in opposite direction the value where the distance of indices is the largest is the hardest value

        with pairing_access_lock:
            if idx in self.positive_embedd_pairs.keys() and idx in self.positive_gt_pairs.keys():
                # intersect the pairs arrays
                possible_pos_partners = np.stack(
                    np.intersect1d(self.positive_gt_pairs[idx], self.positive_embedd_pairs[idx], return_indices=True, assume_unique=True)).T
                # if there is an intersection find the hardest pair
                if possible_pos_partners.shape[0] > 0:
                    # get the differences in indices
                    pos_diffs = possible_pos_partners[:, 1] - possible_pos_partners[:, 2]
                    # get the index of the hardest value
                    max_diff_indices = np.argsort(pos_diffs)[:leeway]
                    positive_indices = possible_pos_partners[max_diff_indices, 0]
                else:
                    positive_indices = np.array([idx])
            else:
                # return the anchor so there is not difference and therefore no loss
                positive_indices = np.array([idx])

            if idx in self.negative_embedd_pairs.keys() and idx in self.negative_gt_pairs.keys():
                possible_neg_partners = np.stack(
                    np.intersect1d(self.negative_gt_pairs[idx], self.negative_embedd_pairs[idx], return_indices=True, assume_unique=True)).T
                if possible_neg_partners.shape[0] > 0:
                    neg_diffs = possible_neg_partners[:, 1] - possible_neg_partners[:, 2]
                    max_diff_indices = np.argsort(neg_diffs)[-leeway:]
                    negative_indices = possible_neg_partners[max_diff_indices, 0]
                else:
                    negative_indices = np.array(self.negative_gt_pairs[idx][0])

            else:
                negative_indices = np.array(self.negative_gt_pairs[idx][0])

        min_samples_found = min([len(positive_indices), len(negative_indices)])

        anchor_indices = np.array([idx] * min_samples_found)
        positive_indices = positive_indices[:min_samples_found]
        negative_indices = negative_indices[:min_samples_found]

        sampled_triplets.extend(np.stack((anchor_indices, positive_indices, negative_indices)).T.tolist())

        sample = torch.stack(
            [self.prott5_embeddings[anchor_indices],
             self.prott5_embeddings[positive_indices],
             self.prott5_embeddings[negative_indices]], dim=1
        )

        return sample

    def simple_sampling(self, idx, leeway):
        """
        This method returns a triplet sampled from the precomputed pairs
        Args:
            leeway:
            idx:

        Returns:

        """
        # Main idea
        # Given an index the arrays that contain the indices for potential postives are fetched
        # The indices in the GT array are sorted ascendingly, the indices in the EMBEDD array are sorted descendingly
        # The intersection between the 2 arrays is computed with np.1dintersect [intersecting_value,index in arr1, index in arr2]
        # Since the arrays are sorted in opposite direction the value where the distance of indices is the largest is the hardest value

        if idx in self.positive_gt_pairs.keys():
            potential_positive_indices = self.positive_gt_pairs[idx]
            positive_embedd_distances = self.embedding_space_distances[idx][potential_positive_indices]
            max_distances = np.argsort(positive_embedd_distances)[-leeway:]

            positive_indices = potential_positive_indices[max_distances]
        else:
            positive_indices = np.array([idx])

        if idx in self.negative_gt_pairs.keys():
            potential_negative_indices = self.negative_gt_pairs[idx]
            negative_embedd_distances = self.embedding_space_distances[idx][potential_negative_indices]
            min_distances = np.argsort(negative_embedd_distances)[:leeway]

            negative_indices = potential_negative_indices[min_distances]

        min_samples_found = min([len(positive_indices), len(negative_indices)])
        anchor_indices = np.array([idx] * min_samples_found)
        positive_indices = positive_indices[:min_samples_found]
        negative_indices = negative_indices[:min_samples_found]

        sampled_triplets.extend(np.stack((anchor_indices, positive_indices, negative_indices)).T.tolist())

        sample = torch.stack(
            [self.prott5_embeddings[anchor_indices],
             self.prott5_embeddings[positive_indices],
             self.prott5_embeddings[negative_indices]], dim=1
        )

        return sample

    def polt_triplet_sampling(self, epoch: int, input_type: Literal['triplets', 'gt'] = 'triplets'):
        pair_01 = Counter()
        pair_02 = Counter()

        if input_type == 'triplets':
            for triplet in sampled_triplets:
                pair_01[(triplet[0], triplet[1])] += 1
                pair_02[(triplet[0], triplet[2])] += 1
        if input_type == 'gt':
            positive_pairs = list(
                itertools.chain(*[[(anchor, positive) for positive in all_positives] for anchor, all_positives in self.positive_gt_pairs.items()]))
            negative_pairs = list(
                itertools.chain(*[[(anchor, negative) for negative in all_negatives] for anchor, all_negatives in self.negative_gt_pairs.items()]))
            for pos_pair in positive_pairs:
                pair_01[(pos_pair[0], pos_pair[1])] += 1
            for neg_pair in negative_pairs:
                pair_02[(neg_pair[0], neg_pair[1])] += 1

        size = len(self.ids)

        heatmap_01 = np.zeros((size, size))
        heatmap_02 = np.zeros((size, size))

        # Populate the heatmap matrices
        for (i, j), count in pair_01.items():
            heatmap_01[i, j] = count
        for (i, j), count in pair_02.items():
            heatmap_02[i, j] = count

        # Plot the heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        sns.heatmap(heatmap_01, cmap='inferno', ax=axs[0])
        axs[0].set_title(f'Heatmap of {input_type} Pair positives {epoch}')

        sns.heatmap(heatmap_02, cmap='inferno', ax=axs[1])
        axs[1].set_title(f'Heatmap of {input_type} Pair Negatives {epoch}')

        plt.show()

    def serialize_for_storage(self):
        self.positive_embedd_pairs = {}
        self.negative_embedd_pairs = {}
        self.embedding_space_distances = None

    def set_shared_resources(self):
        self.positive_embedd_pairs = pos_embedd_pairings
        self.negative_embedd_pairs = neg_embedd_pairings
        self.embedding_space_distances = np.frombuffer(embedding_distances.get_obj(), dtype=np.float64).reshape(self.cophentic_distances.shape)

    def __len__(self):
        return self.cophentic_distances.shape[0]
