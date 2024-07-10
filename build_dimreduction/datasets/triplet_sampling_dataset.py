import itertools
import logging.config
import multiprocessing
import os.path
from bisect import bisect
from collections import Counter
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ete4 import Tree
from torch.utils.data import Dataset

from build_dimreduction.utils.ProtSeqEmbedder import ProtSeqEmbedder
from build_dimreduction.utils.seeding import get_input_data, zscore_normalize
from evaluation_visualization.utils import convert_to_full
from evaluation_visualization.tree_building import TreeBuilder

from build_dimreduction.utils.triplet_mining import compute_pos_neg_pairs, multi_embedding_distances, neg_embedd_pairings, pos_embedd_pairings, \
    sampled_triplets, pairing_access_locks, manager

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


class TripletSamplingDataset(Dataset):
    """
    This dataset creates triplets given a groundtruth matrix
    """


    # TODO switch to input of tree data not distance matrix work with gt trees not distances
    def __init__(self, model: Literal['prott5', 'esm'], paths_to_input_data: list[str], device):
        self.ids = {}
        self.prott5_embeddings = {}
        self.cophentic_distances = {}
        self.dataset_embedding_distances = {}
        self.positive_embedd_pairs = pos_embedd_pairings
        self.negative_embedd_pairs = neg_embedd_pairings

        self.id_lengths = [0]
        self.data_names = []
        for input_data_path in paths_to_input_data:
            data_name = input_data_path.split('/')[-1]
            # get paths of input files
            path_to_fasta, path_to_gt_newick = get_input_data(input_data_path)
            assert os.path.isfile(path_to_fasta), "Path to fasta does not exist"
            assert os.path.isfile(path_to_gt_newick), "Path to distances does not exist"
            # init embedder
            embedder = ProtSeqEmbedder(model)
            # get embedding data
            embedd_data = zip(*embedder.get_raw_embeddings(path_to_fasta))
            seq_ids, prott5_embedds = list(embedd_data)

            # set ids and embeddings
            self.prott5_embeddings[data_name] = torch.stack(prott5_embedds)
            self.ids[data_name] = seq_ids

            # set the cophentic distances
            cophentic_dists = self._get_cophentic_distmatrix(path_to_gt_newick, data_name)
            cophentic_dists = zscore_normalize(cophentic_dists)
            self.cophentic_distances[data_name] = cophentic_dists

            # initialize shared dicts and arrays for the data name
            multi_embedding_distances[data_name] = multiprocessing.Array('d', cophentic_dists.shape[0] ** 2)
            pos_embedd_pairings[data_name] = manager.dict()
            neg_embedd_pairings[data_name] = manager.dict()
            pairing_access_locks[data_name] = manager.Lock()

            # define numpy array on shared array
            self.dataset_embedding_distances[data_name] = np.frombuffer(multi_embedding_distances[data_name].get_obj(), dtype=np.float64).reshape(
                cophentic_dists.shape)

            self.id_lengths.append(int(cophentic_dists.shape[0]) + self.id_lengths[-1])
            self.data_names.append(data_name)
            sampled_triplets[data_name] = manager.list()

        self.device = device

        self.leeway = None

        self.positive_threshold = None
        self.negative_threshold = None

        self.positive_gt_pairs = {}
        self.negative_gt_pairs = {}

        self.plot_distance_maps(distance_type='cophentic')

    def set_gt_pairings(self):
        """
        Sets the pairs that will be considered as ground truth postives or neagtives during training
        Returns:
            None, Updates the instance variables

        """

        for key in self.cophentic_distances.keys():
            positive_condition = ((self.cophentic_distances[key] < self.positive_threshold) & (self.cophentic_distances[key] > 0))
            negative_condition = self.cophentic_distances[key] > self.negative_threshold

            self.positive_gt_pairs[key], self.negative_gt_pairs[key] = compute_pos_neg_pairs(distance_matrix=self.cophentic_distances[key],
                                                                                             positive_condition=positive_condition,
                                                                                             negative_condition=negative_condition,
                                                                                             desc=False)

    def set_constants(self, pos_threshold: float, neg_threshold: float, leeway: int):
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

    def _get_cophentic_distmatrix(self, path_to_gt_tree, data_name) -> np.ndarray:
        """
        Computes a cophentic distance matrix given a path to gt tree
        Args:
            path_to_gt_tree: The path to the gt tree in the newick format

        Returns:
            A dataframe with sequence names as index and columns and the respective distances
        """
        # build the tree
        logger.debug(f"Loading gt tree from {path_to_gt_tree}")
        treebuilder = TreeBuilder(path_to_tree_data=path_to_gt_tree, is_truth=True)
        cophentic_distances, names = treebuilder.get_cophentic_distances()

        return pd.DataFrame(cophentic_distances, index=names, columns=names).reindex(self.ids[data_name], axis=0).reindex(self.ids[data_name],
                                                                                                                          axis=1).values

    def __getitem__(self, idx):

        insertion_index = bisect(self.id_lengths, idx)
        data_name = self.data_names[insertion_index - 1]
        data_name_index = idx - self.id_lengths[insertion_index - 1]
        return self.simple_sampling(data_name_index, self.leeway, data_name)

    def benjamin_sampling(self, idx, leeway, data_name):
        """
        This method returns a triplet sampled from the precomputed pairs
        Args:
            data_name:
            leeway:
            idx:

        Returns:

        """
        # Main idea
        # Given an index the arrays that contain the indices for potential postives are fetched
        # The indices in the GT array are sorted ascendingly, the indices in the EMBEDD array are sorted descendingly
        # The intersection between the 2 arrays is computed with np.1dintersect [intersecting_value,index in arr1, index in arr2]
        # Since the arrays are sorted in opposite direction the value where the distance of indices is the largest is the hardest value

        with pairing_access_locks[data_name]:
            if idx in self.positive_embedd_pairs[data_name].keys() and idx in self.positive_gt_pairs[data_name].keys():
                # intersect the pairs arrays
                possible_pos_partners = np.stack(
                    np.intersect1d(self.positive_gt_pairs[data_name][idx], self.positive_embedd_pairs[data_name][idx], return_indices=True,
                                   assume_unique=True)).T
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

            if idx in self.negative_embedd_pairs[data_name].keys() and idx in self.negative_gt_pairs[data_name].keys():
                possible_neg_partners = np.stack(
                    np.intersect1d(self.negative_gt_pairs[data_name][idx], self.negative_embedd_pairs[data_name][idx], return_indices=True,
                                   assume_unique=True)).T
                if possible_neg_partners.shape[0] > 0:
                    neg_diffs = possible_neg_partners[:, 1] - possible_neg_partners[:, 2]
                    max_diff_indices = np.argsort(neg_diffs)[-leeway:]
                    negative_indices = possible_neg_partners[max_diff_indices, 0]
                else:
                    negative_indices = np.array([self.negative_embedd_pairs[data_name][idx][-1]])

            else:
                negative_indices = np.array([self.negative_embedd_pairs[data_name][idx][-1]])

        try:
            test_len = len(negative_indices)
        except TypeError:
            print('hi')

        min_samples_found = min([len(positive_indices), len(negative_indices)])

        anchor_indices = np.array([idx] * min_samples_found)
        positive_indices = positive_indices[:min_samples_found]
        negative_indices = negative_indices[:min_samples_found]

        sampled_triplets[data_name].extend(np.stack((anchor_indices, positive_indices, negative_indices)).T.tolist())

        sample = torch.stack(
            [self.prott5_embeddings[data_name][anchor_indices],
             self.prott5_embeddings[data_name][positive_indices],
             self.prott5_embeddings[data_name][negative_indices]], dim=1
        )

        return sample

    def simple_sampling(self, idx, leeway, data_name):
        """
        This method returns a triplet sampled from the precomputed pairs
        Args:
            data_name:
            leeway:
            idx:

        Returns:

        """
        # Main idea
        # Given an index the arrays that contain the indices for potential postives are fetched
        # The indices in the GT array are sorted ascendingly, the indices in the EMBEDD array are sorted descendingly
        # The intersection between the 2 arrays is computed with np.1dintersect [intersecting_value,index in arr1, index in arr2]
        # Since the arrays are sorted in opposite direction the value where the distance of indices is the largest is the hardest value

        if idx in self.positive_gt_pairs[data_name].keys():
            potential_positive_indices = self.positive_gt_pairs[data_name][idx]
            positive_embedd_distances = self.dataset_embedding_distances[data_name][idx][potential_positive_indices]
            max_distances = np.argsort(positive_embedd_distances)[-leeway:]

            positive_indices = potential_positive_indices[max_distances]
        else:
            positive_indices = np.array([idx])

        if idx in self.negative_gt_pairs[data_name].keys():
            potential_negative_indices = self.negative_gt_pairs[data_name][idx]
            negative_embedd_distances = self.dataset_embedding_distances[data_name][idx][potential_negative_indices]
            min_distances = np.argsort(negative_embedd_distances)[:leeway]

            negative_indices = potential_negative_indices[min_distances]
        else:
            negative_indices = np.array([self.negative_embedd_pairs[data_name][idx][-1]])

        min_samples_found = min([len(positive_indices), len(negative_indices)])
        anchor_indices = np.array([idx] * min_samples_found)
        positive_indices = positive_indices[:min_samples_found]
        negative_indices = negative_indices[:min_samples_found]

        sampled_triplets[data_name].extend(np.stack((anchor_indices, positive_indices, negative_indices)).T.tolist())

        sample = torch.stack(
            [self.prott5_embeddings[data_name][anchor_indices],
             self.prott5_embeddings[data_name][positive_indices],
             self.prott5_embeddings[data_name][negative_indices]], dim=1
        )

        return sample

    def polt_triplet_sampling(self, epoch: int, input_type: Literal['triplets', 'gt'] = 'triplets'):

        for data_name in sampled_triplets.keys():
            pair_pos_sampled = Counter()
            pair_neg_sampled = Counter()

            pair_pos_gt = Counter()
            pair_neg_gt = Counter()

            for triplet in sampled_triplets[data_name]:
                pair_pos_sampled[(triplet[0], triplet[1])] += 1
                pair_neg_sampled[(triplet[0], triplet[2])] += 1

            positive_pairs = list(
                itertools.chain(
                    *[[(anchor, positive) for positive in all_positives] for anchor, all_positives in self.positive_gt_pairs[data_name].items()]))
            negative_pairs = list(
                itertools.chain(
                    *[[(anchor, negative) for negative in all_negatives] for anchor, all_negatives in self.negative_gt_pairs[data_name].items()]))
            for pos_pair in positive_pairs:
                pair_pos_gt[(pos_pair[0], pos_pair[1])] += 1
            for neg_pair in negative_pairs:
                pair_neg_gt[(neg_pair[0], neg_pair[1])] += 1

            size = len(self.ids[data_name])

            heatmap_pos_sampled = np.zeros((size, size))
            heatmap_neg_sampled = np.zeros((size, size))

            heatmap_pos_gt = np.zeros((size, size))
            heatmap_neg_gt = np.zeros((size, size))

            # Populate the heatmap matrices
            for (i, j), count in pair_pos_sampled.items():
                heatmap_pos_sampled[i, j] = count
            for (i, j), count in pair_neg_sampled.items():
                heatmap_neg_sampled[i, j] = count

            for (i, j), count in pair_pos_gt.items():
                heatmap_pos_gt[i, j] = count
            for (i, j), count in pair_neg_gt.items():
                heatmap_neg_gt[i, j] = count

            # Plot the heatmaps
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))

            sns.heatmap(heatmap_pos_sampled, cmap='inferno', ax=axs[0, 0])
            axs[0, 0].set_title('Heatmap of sampled positives')

            sns.heatmap(heatmap_neg_sampled, cmap='inferno', ax=axs[0, 1])
            axs[0, 1].set_title('Heatmap of sampled negatives')

            sns.heatmap(heatmap_pos_gt, cmap='inferno', ax=axs[1, 0])
            axs[1, 0].set_title('Heatmap of gt positives')

            sns.heatmap(heatmap_neg_gt, cmap='inferno', ax=axs[1, 1])
            axs[1, 1].set_title('Heatmap of gt negatives')

            fig.suptitle(f'{data_name} epoch: {epoch}')
            fig.tight_layout()
            plt.show()

    def serialize_for_storage(self):
        self.positive_embedd_pairs = {}
        self.negative_embedd_pairs = {}
        self.dataset_embedding_distances = {}

    def set_shared_resources(self):
        self.positive_embedd_pairs = pos_embedd_pairings
        self.negative_embedd_pairs = neg_embedd_pairings

        for data_name in self.ids.keys():
            self.dataset_embedding_distances[data_name] = np.frombuffer(multi_embedding_distances[data_name].get_obj(), dtype=np.float64).reshape(
                self.cophentic_distances[data_name].shape)

    def plot_distance_maps(self, distance_type: Literal['cophentic', 'embedd'], mode: Literal['distances', 'dist'] = 'distances', epoch: int = None):

        if distance_type == 'cophentic':
            map_type = self.cophentic_distances
        elif distance_type == 'embedd':
            map_type = self.dataset_embedding_distances

        fig, axs = plt.subplots(1, len(map_type), figsize=(15, 5))
        for i, key in enumerate(map_type.keys()):
            if mode == 'distances':
                sns.heatmap(map_type[key], ax=axs[i])
            elif mode == 'dist':
                sns.histplot(map_type[key].flatten(), ax=axs[i])
                axs[i].axvline(x=self.positive_threshold, color='r', linestyle='--')
                axs[i].axvline(x=self.negative_threshold, color='r', linestyle='--')

            if epoch is not None:
                axs[i].set_title(f'{key} epoch:{epoch}')
            else:
                axs[i].set_title(f'{key}')

        plt.show()

    def __len__(self):
        return sum([len(ids) for ids in self.ids.values()])
