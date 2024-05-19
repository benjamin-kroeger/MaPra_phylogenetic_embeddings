import hashlib
import logging.config
import os
from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import treeswift
# import umap.umap_ as umap
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from ete4 import Tree
from ete4.treeview import TreeStyle, NodeStyle
from scipy.spatial import distance

# from umap import plot

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def _square_to_lower_triangle(dist_square):
    n = len(dist_square)
    dist_lower_triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(dist_square[i][j])
        dist_lower_triangle.append(row)
    return dist_lower_triangle


class DistmapVizClust:

    family_map = defaultdict()
    def __init__(self, distmap: pd.DataFrame, is_truth: bool, output_file_suffix: str = '') -> None:
        self.distmap = distmap.values
        self.out_suffix = 'truth' if is_truth else 'pred' + output_file_suffix
        self.names = list(distmap.columns)

        assert len(self.names) == distmap.shape[0] == distmap.shape[1]

    def get_tree(self, method: Literal['upgma', 'nj'] = 'upgma') -> tuple[np.array, Tree]:
        """
        Computes the the tree from precomputed distances and saves the tree to disk. It also
        computes and returns the cophenetic distance matrix.
        Args:
            method: The tree building method to use.

        Returns:
            Cophentic distance matrix.
        """

        out_filepath = os.path.join(os.environ['OUTPUT_DIR'], f'%s_{method}_{self.out_suffix}')
        tree_data = self._compute_tree(out_filepath, method)
        tree = self._draw_tree(out_filepath, tree_data)

        return np.array(tree.cophenetic_matrix()[0]), tree

    def _compute_tree(self, out_filepath: str, method: Literal['upgma', 'nj'] = 'upgma') -> str:
        """
        Compute a tree on the distance matrix using the specified method.
        Args:
            method:

        Returns:
            The newick string of the tree.
        """
        # create a tree
        triag_dist = DistanceMatrix(matrix=_square_to_lower_triangle(self.distmap), names=self.names)
        constructor = DistanceTreeConstructor()
        logger.debug(f'Constructing {method} tree')
        if method == 'upgma':
            tree_data = constructor.upgma(distance_matrix=triag_dist)
        elif method == 'nj':
            tree_data = constructor.nj(distance_matrix=triag_dist)
        # store the newick file
        newick_filepath = out_filepath % 'Treedata' + '.nw'
        logger.debug(f'Writing newick to {newick_filepath}')
        with open(newick_filepath, 'w') as file:
            file.write(tree_data.format("newick"))

        return tree_data.format("newick")

    def _draw_tree(self, out_filepath: str, tree_data: str) -> Tree:
        """
        Draw the tree using ete4 and save the tree to disk.
        Args:
            out_filepath: The path where to save it to
            tree_data: The tree in a newick format

        Returns:
            None
        """
        # draw the tree
        t = Tree(tree_data.format("newick"), parser=1)
        circular_style = TreeStyle()
        circular_style.show_leaf_name = False
        circular_style.mode = 'c'  # draw tree in circular mode
        circular_style.scale = 20
        for node in t.traverse():
            self._color_leaf_by_fam(node)
        # Write the tree to output
        tree_vis_path = out_filepath % 'Tree' + '.png'
        logger.debug(f'Writing tree to {tree_vis_path}')
        t.render(tree_vis_path, w=183, units='mm', tree_style=circular_style)

        return t

    def _color_leaf_by_fam(self, node):
        if node.is_leaf:
            node.img_style["fgcolor"] = '#' + hashlib.sha1(node.name.split('_')[-1].encode('utf-8')).hexdigest()[:6]
            node.img_style["shape"] = "square"  # Red for protein family A
        else:
            node.img_style["fgcolor"] = "#000000"
            node.img_style["size"] = 1



    def get_umap(self) -> np.array:
        out_filepath = os.path.join(os.environ['OUTPUT_DIR'], f'Umap_{self.out_suffix}')

        umap_calc = umap.UMAP(n_components=2, random_state=42, densmap=True, n_jobs=8,
                              metric='precomputed',
                              output_metric='euclidean')
        umap_embedding = umap_calc.fit(self.distmap)
        plot.points(umap_embedding)
        plt.savefig(out_filepath)
        distances_umap = distance.cdist(umap_embedding.embedding_, umap_embedding.embedding_, metric='euclidean')
        return distances_umap, umap_embedding

    def draw_tsne(self):
        pass
