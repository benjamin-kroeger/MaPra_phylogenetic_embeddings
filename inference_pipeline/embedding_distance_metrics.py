import numpy as np
import torch
from scipy.spatial import distance


class sim_scorer:
    """
    This class provides multiple methods to calculate similarity between two prott5_embeddings.
    """

    @staticmethod
    def cosine_similarity(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
        assert embedding1.shape[1] == embedding2.shape[1], "The embedding dimensions do not match"
        return distance.cdist(embedding1, embedding2, metric='cosine')

    @staticmethod
    def euclidean_distance(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
        assert embedding1.shape[1] == embedding2.shape[1], "The embedding dimensions do not match"
        return distance.cdist(embedding1, embedding2, metric='euclidean')

    @staticmethod
    def manhattan(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
        assert embedding1.shape[1] == embedding2.shape[1], "The embedding dimensions do not match"
        return distance.cdist(embedding1, embedding2, metric='cityblock')

    @staticmethod
    def jensenshannon(embedding1: torch.tensor, embedding2: torch.tensor) -> torch.tensor:
        assert embedding1.shape[1] == embedding2.shape[1], "The embedding dimensions do not match"
        return distance.cdist(embedding1, embedding2, metric='jensenshannon')
