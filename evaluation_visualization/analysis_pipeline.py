import logging.config

import numpy as np
import pandas as pd

from evaluation_visualization.compute_distmap_metrics import DistmapMetrics
from evaluation_visualization.tree_building import TreeBuilder
from evaluation_visualization.utils import _check_triangularitry, convert_to_full
from inference_pipeline.embedding_distance_metrics import sim_scorer

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def align_dfs(df1, df2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns two dataframes. Make sure the columns and indices are aligned.
    Args:
        df1: DF1
        df2: DF2

    Returns:
        2 aligned dataframes
    """
    if _check_triangularitry(df1):
        df1 = convert_to_full(df1)
    if _check_triangularitry(df2):
        df2 = convert_to_full(df2)

    df1 = df1.sort_index(axis=0).sort_index(axis=1)
    df2 = df2.sort_index(axis=0).sort_index(axis=1)

    # perform assertions
    assert (df1.columns == df1.index).all(), "The cols and index of df1 do not match"
    assert (df2.columns == df2.index).all(), "The cols and index of df2 do not match"
    assert (df1.columns == df2.columns).all() and (df1.index == df2.index).all(), "There is a column mismatch"

    return df1, df2


def analyse_distmaps(distmap1_pred: pd.DataFrame, distmap2_truth: pd.DataFrame):
    """
    Given 2 distance matrices, run a series of comparisons on the 2
    Args:
        distmap1_pred:
        distmap2_truth:

    Returns:
        None
    """
    from evaluation_visualization.clustering import get_umap
    # align the dfs for proper comparisson
    distmap1_pred, distmap2_truth = align_dfs(distmap1_pred, distmap2_truth)
    logger.debug('Initializing distmap visualization')
    # create Trees for each dist matrix
    distmap_visclust1 = TreeBuilder(distmap1_pred, is_truth=False)
    distmap_visclust2 = TreeBuilder(distmap2_truth, is_truth=True)

    # geth umap
    get_umap(distmap1_pred)

    logger.debug('Visualizing and scoring new representations')
    clustering_results = [
        ('nj', distmap_visclust1.get_tree(), distmap_visclust2.get_tree()), ]

    logger.debug('Computing distmap comparison metrics')
    metrics = []
    labels = []
    # run some numerical tests on the 2 distance matrices
    for method_name, pred, truth in clustering_results:
        metrics.append({
            "trustworthiness": DistmapMetrics.compute_trustworthiness(pred[0], truth[0]),
            "spearman": DistmapMetrics.compute_spearman(pred[0], truth[0]),
            "norm_robinson": DistmapMetrics.compute_norm_robinson(pred[1], truth[1]),
        })
        labels.append(method_name)

    # add metric on the raw distance matrices
    labels.append("raw_distances")
    metrics.append({
        "trustworthiness": DistmapMetrics.compute_trustworthiness(distmap1_pred.values, distmap2_truth.values),
        "spearman": DistmapMetrics.compute_spearman(distmap1_pred.values, distmap2_truth.values),
        "norm_robinson": pd.NA,
    })

    summary = pd.DataFrame(metrics, index=labels)
    print(summary)


if __name__ == '__main__':
    np.random.seed(42)
    test_embedds1 = np.random.rand(50, 50)
    test_embedds2 = np.random.rand(50, 50)

    test_dist1 = sim_scorer.euclidean_distance(test_embedds1, test_embedds1)
    test_dist2 = sim_scorer.euclidean_distance(test_embedds2, test_embedds2)

    analyse_distmaps(test_dist1, test_dist2)
