import numpy as np
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from ete4 import Tree


class DistmapMetrics:

    @staticmethod
    def compute_trustworthiness(distmap_truth: np.array, distmap_pred: np.array) -> float:
        """
        Computes the trustworthiness between two distance maps used to create trees
        Args:
            distmap_truth:
            distmap_pred:

        Returns:
            The trustworthiness score between [0,1]
        """
        assert distmap_truth.shape[0] == distmap_truth.shape[1], "The truth distance map is not square"

        return trustworthiness(X=distmap_truth, X_embedded=distmap_pred, n_neighbors=10, metric='precomputed')

    @staticmethod
    def compute_spearman(distmap_truth: np.array, distmap_pred: np.array) -> float:
        """
        Computes the spearman's correlation between two distance maps used to create trees
        Args:
            distmap_truth:
            distmap_pred:

        Returns:
            The spearmann rank correlation
        """

        return spearmanr(squareform(distmap_truth, checks=False), squareform(distmap_pred, checks=False))[0]

    @staticmethod
    def compute_norm_robinson(tree_truth:Tree, tree_pred: Tree) -> float:

        rf_res = tree_truth.robinson_foulds(tree_pred,unrooted_trees=True)

        return rf_res[0] / rf_res[1]

    @staticmethod
    def compute_edge_similarity(tree_truth:Tree, tree_pred: Tree) -> float:
        pass



    @staticmethod
    def compute_dup_aware_distance(distmap_truth: np.array, distmap_pred: np.array) -> float:
        # TODO: Use ete TreeKO method
        pass
