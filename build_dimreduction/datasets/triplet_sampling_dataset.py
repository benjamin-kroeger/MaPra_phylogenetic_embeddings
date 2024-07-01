import logging.config
import os.path
from typing import Literal
import seaborn as sns
import matplotlib.pyplot as plt
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
from collections import Counter
import multiprocessing
logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)



# Initialize sampled_triplets as a multiprocessing list outside the class
manager = multiprocessing.Manager()
sampled_triplets = manager.list()


def _sort_and_group_pairings(pairs: tuple[np.ndarray, np.ndarray], distances: np.ndarray, desc: bool = False) -> dict[int:np.ndarray, np.ndarray]:
    """
    Given index pairings and the corresponding distance matrix this method sorts the pairs in the specified order and groups them

    Args:
        pairs: A tuple of Arrays with the value at the respective indices being one pair
        distances: A np array with the distances
        desc: Whether to sort the pairs from smallest to largest or vice versa

    Returns:

    """

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

    # sort all distances along columns
    # if we want to reverse the sorting order each value is negated
    if desc:
        distances = -distances
    # sort the distances along each row and get the indices of the values
    ordered_dists = distances.argsort(axis=1)

    # The main idea here is that instead of comparing the value of selected pairs
    # the presorted indices are used. All indices in the argsorted array that are not in the pairs array are dropped
    # The order is preserved while also filtering out "pairs" that are not identified with the set threshold

    # update order for each entry
    for x in pairs_dict.keys():
        # get indices where the index from the distmatrix is in the pairing index list
        # only keep these indices from the order dist indices
        ordered_indices = ordered_dists[x][np.where(np.isin(ordered_dists[x], pairs_dict[x]))]
        pairs_dict[x] = ordered_indices

    return pairs_dict


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
        self.embedding_space_distances = None

        ax = sns.heatmap(self.cophentic_distances)
        ax.set_title('Distance Matrix Truth')
        plt.show()

        self.positive_threshold = None
        self.negative_threshold = None

        self.positive_gt_pairs = None
        self.negative_gt_pairs = None

        self.positive_embedd_pairs = None
        self.negative_embedd_pairs = None


    def _compute_pos_neg_pairs(self, distance_matrix: np.array, positive_condition: np.array, negative_condition: np.array, desc=False) -> tuple[
        dict, dict]:
        """
        Computes ordered pairs of positives and negatives given a distance matrix and conditions
        Args:
            distance_matrix: The matrix on which to compute
            positive_condition: A condition for what qualifies as a positive
            negative_condition: A condition for what qualifies as a negative
            desc:

        Returns:
            A tuple of dicts with ordered pairings
        """

        postive_pairs = np.where(positive_condition)
        negative_pairs = np.where(negative_condition)

        pos_pairings = _sort_and_group_pairings(postive_pairs, distance_matrix, desc=desc)
        neg_pairings = _sort_and_group_pairings(negative_pairs, distance_matrix, desc=desc)

        return pos_pairings, neg_pairings

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

        self.positive_gt_pairs, self.negative_gt_pairs = self._compute_pos_neg_pairs(distance_matrix=self.cophentic_distances,
                                                                                     positive_condition=positive_condition,
                                                                                     negative_condition=negative_condition,
                                                                                     desc=False)

    def compute_embedding_distances(self, model_forward):
        # compute distances on all embeddings
        phy_embedds = []
        for prott5_embedd in self.prott5_embeddings:
            phy_embedds.append(model_forward(prott5_embedd.to(self.device)))

        norm_embeddings = F.normalize(torch.stack(phy_embedds), p=2, dim=1)
        embedding_space_dist = np.abs(1 - np.abs(torch.mm(norm_embeddings, norm_embeddings.t()).data.cpu().numpy()))
        return embedding_space_dist

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

        embedding_space_dist = self.compute_embedding_distances(model_forward)
        self.embedding_space_distances = embedding_space_dist

        positive_condition = np.ones(shape=embedding_space_dist.shape, dtype=bool)
        np.fill_diagonal(positive_condition, False)

        # Order gt pairings in ascending order
        # Positives: Most similar has largest index
        # Negatives: Most different has largest index
        self.positive_embedd_pairs, self.negative_embedd_pairs = self._compute_pos_neg_pairs(distance_matrix=embedding_space_dist,
                                                                                             positive_condition=positive_condition,
                                                                                             negative_condition=np.ones(
                                                                                                 shape=embedding_space_dist.shape, dtype=bool),
                                                                                             desc=False)

    def set_thresholds(self, pos_threshold: float, neg_threshold: float):
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
        """
        This method returns a triplet sampled from the precomputed pairs
        Args:
            idx:

        Returns:

        """
        # Main idea
        # Given an index the arrays that contain the indices for potential postives are fetched
        # The indices in the GT array are sorted ascendingly, the indices in the EMBEDD array are sorted descendingly
        # The intersection between the 2 arrays is computed with np.1dintersect [intersecting_value,index in arr1, index in arr2]
        # Since the arrays are sorted in opposite direction the value where the distance of indices is the largest is the hardest value

        leway = 15

        if idx in self.positive_embedd_pairs.keys() and idx in self.positive_gt_pairs.keys():
            # intersect the pairs arrays
            possible_pos_partners = np.stack(
                np.intersect1d(self.positive_gt_pairs[idx], self.positive_embedd_pairs[idx], return_indices=True, assume_unique=True)).T
            # if there is an intersection find the hardest pair
            if possible_pos_partners.shape[0] > 0:
                # get the differences in indices
                pos_diffs = possible_pos_partners[:, 1] - possible_pos_partners[:, 2]
                # get the index of the hardest value
                max_diff_indices = np.argsort(pos_diffs)[:leway]
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
                max_diff_indices = np.argsort(neg_diffs)[-leway:]
                negative_indices = possible_neg_partners[max_diff_indices, 0]
            else:
                negative_indices = np.array(self.negative_gt_pairs[idx][0])

        else:
            negative_indices = np.array(self.negative_gt_pairs[idx][0])

        min_samples_found = min([len(positive_indices),len(negative_indices)])

        anchor_indices = np.array([idx]*min_samples_found)
        positive_indices = positive_indices[:min_samples_found]
        negative_indices = negative_indices[:min_samples_found]

        sampled_triplets.extend(np.stack((anchor_indices,positive_indices,negative_indices)).T.tolist())

        sample = torch.stack(
            [self.prott5_embeddings[anchor_indices],
             self.prott5_embeddings[positive_indices],
             self.prott5_embeddings[negative_indices]], dim=1
        )

        return sample

    def polt_triplet_sampling(self):
        pair_01 = Counter()
        pair_02 = Counter()

        for triplet in sampled_triplets:
            pair_01[(triplet[0], triplet[1])] += 1
            pair_02[(triplet[0], triplet[2])] += 1

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

        sns.heatmap(heatmap_01, ax=axs[0])
        axs[0].set_title('Heatmap of Pair positives')

        sns.heatmap(heatmap_02, ax=axs[1])
        axs[1].set_title('Heatmap of Pair Negatives')

        plt.show()

    def __len__(self):
        return self.cophentic_distances.shape[0]
