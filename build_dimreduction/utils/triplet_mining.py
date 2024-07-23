import multiprocessing

import numpy as np
import torch
import torch.nn.functional as F

# Initialize sampled_triplets as a multiprocessing list outside the class
manager = multiprocessing.Manager()
sampled_triplets = {}  # stores lists with all the triplets that were sampled from one dataset
multi_embedding_distances = {}  # stores arrays for the embeddings distances
pos_embedd_pairings = {}  # stores dicts with sorted indices based on distance
neg_embedd_pairings = {}  # stores dicts with sorted indices based on distance
pairing_access_locks = {} # stores locks


def compute_embedding_distances(embeddings_tensor, model_forward, model_device) -> np.ndarray:
    """
    Given an embedding tensor and a model forward function, compute the pairwise distances
    Args:
        embeddings_tensor: A tensor with all embeddings
        model_forward: The model forward function
        model_device: The device on which the tensor and model are

    Returns:
        A pairwise distance matrix
    """
    # compute distances on all embeddings
    phy_embedds = []
    batch_size = 100
    for index in range(0, embeddings_tensor.shape[0], batch_size):
        phy_embedds.append(model_forward(embeddings_tensor[index:index + batch_size, :].to(model_device)))

    norm_embeddings = F.normalize(torch.cat(phy_embedds, dim=0), p=2, dim=1)
    # embedding_space_dist = 1 - F.relu(torch.mm(norm_embeddings, norm_embeddings.t())).data.cpu().numpy() # any value lower 0 is 0
    embedding_space_dist = 1 - torch.mm(norm_embeddings, norm_embeddings.t()).data.cpu().numpy()  # allow for more range in similarity
    return embedding_space_dist


def sort_and_group_pairings(pairs: tuple[np.ndarray, np.ndarray], distances: np.ndarray, desc: bool = False) -> dict[int:np.ndarray, np.ndarray]:
    """
    Given index pairings and the corresponding distance matrix this method sorts the pairs in the specified order and groups them

    Args:
        pairs: A tuple of Arrays with the value at the respective indices being one pair
        distances: A np array with the distances
        desc: Whether to sort the pairs from smallest to largest or vice versa

    Returns:
        A dict with the seq indices as keys and an array with all pos/neg indices
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

    return pairs_dict  #


def compute_pos_neg_pairs(distance_matrix: np.array, positive_condition: np.array, negative_condition: np.array, desc=False) -> tuple[
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

    pos_pairings = sort_and_group_pairings(postive_pairs, distance_matrix, desc=desc)
    neg_pairings = sort_and_group_pairings(negative_pairs, distance_matrix, desc=desc)

    return pos_pairings, neg_pairings


def set_per_dataname_pairings(data_name, embedding_tensor, model_forward, device):
    """
    This method computes positive and negative pairing embeddings for all pairings.
    1. First all embeddings are generated
    2. The cosine distance is computed across all embeddings
    3. The thresholds for want counts as pos and neg are computed
    4. The distances and thresholds are passed of to pair finding
    Args:
        data_name: The dataset on which to compute pairings
        embedding_tensor:
        model_forward: A forward funtion of a model


    """

    global multi_embedding_distances
    global pos_embedd_pairings
    global neg_embedd_pairings
    global pairing_access_locks
    embedding_space_dist = compute_embedding_distances(embeddings_tensor=embedding_tensor, model_forward=model_forward, model_device=device)

    # say that all distances are eligible to be considered
    positive_condition = np.ones(shape=embedding_space_dist.shape, dtype=bool)
    # avoid self distances
    np.fill_diagonal(positive_condition, False)

    pos, neg = compute_pos_neg_pairs(distance_matrix=embedding_space_dist,
                                     positive_condition=positive_condition,
                                     negative_condition=np.ones(shape=embedding_space_dist.shape, dtype=bool),
                                     desc=False)

    # set the pairings and distances
    with pairing_access_locks[data_name]:
        # update the embedding distances used by the dataset, might be wrong
        np.frombuffer(multi_embedding_distances[data_name].get_obj(), dtype=np.float64)[:] = embedding_space_dist.ravel()

        pos_embedd_pairings[data_name].clear()
        pos_embedd_pairings[data_name].update(pos)
        neg_embedd_pairings[data_name].clear()
        neg_embedd_pairings[data_name].update(neg)


def set_embedding_pairings(dataset, model_forward, device):
    """
    Update the distances and pairings for all embeddings
    Args:
        dataset: The dataset whos distances are being updated
        model_forward: The model forward function
        device: The device on which the tensor and model are

    Returns:
        None, updated the shared resources
    """
    for data_name in dataset.prott5_embeddings.keys():
        embedding_tensor = dataset.prott5_embeddings[data_name]
        set_per_dataname_pairings(data_name, embedding_tensor, model_forward, device)
