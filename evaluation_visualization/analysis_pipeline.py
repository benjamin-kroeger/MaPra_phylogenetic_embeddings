import logging.config

import numpy as np
import pandas as pd

from evaluation_visualization.compute_distmap_metrics import DistmapMetrics
from evaluation_visualization.distmap_visualization import DistmapVizClust
from inference_pipeline.embedding_distance_metrics import sim_scorer

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def analyse_distmaps(distmap1: np.array, distmap2: np.array,names:list[str]):
    logger.debug('Initializing distmap visualization')
    distmap_visclust1 = DistmapVizClust(distmap1, is_truth=False, names=names)
    distmap_visclust2 = DistmapVizClust(distmap2, is_truth=True, names=names)

    logger.debug('Visualizing and scoring new representations')
    clustering_results = [  # ('umap', distmap_visclust1.get_umap(), distmap_visclust2.get_umap()),
        ('upgma', distmap_visclust1.get_tree(method='upgma'), distmap_visclust2.get_tree(method='upgma')),
        ('nj', distmap_visclust1.get_tree(method='nj'), distmap_visclust2.get_tree(method='nj')), ]

    logger.debug('Computing distmap comparison metrics')
    metrics = []
    labels = []
    for method_name, pred, truth in clustering_results:
        metrics.append({
            "trustworthiness": DistmapMetrics.compute_trustworthiness(pred[0], truth[0]),
            "spearman": DistmapMetrics.compute_spearman(pred[0], truth[0]),
            "norm_robinson": DistmapMetrics.compute_norm_robinson(pred[1], truth[1]),
        })
        labels.append(method_name)

    summary = pd.DataFrame(metrics, index=labels)
    print(summary)


if __name__ == '__main__':
    np.random.seed(42)
    test_embedds1 = np.random.rand(50, 50)
    test_embedds2 = np.random.rand(50, 50)

    test_dist1 = sim_scorer.euclidean_distance(test_embedds1, test_embedds1)
    test_dist2 = sim_scorer.euclidean_distance(test_embedds2, test_embedds2)

    analyse_distmaps(test_dist1, test_dist2)
