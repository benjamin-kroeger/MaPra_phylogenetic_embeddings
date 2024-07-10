import logging.config
import os
import re
import tempfile
from collections import defaultdict
from subprocess import PIPE, Popen, STDOUT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import umap.umap_ as umap
from ete4 import Tree
from ete4.treeview import TreeStyle, NodeStyle, RectFace, add_face_to_node
from scipy.spatial import distance
from evaluation_visualization.utils import get_color

# from umap import plot

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)
scalar = 20


def layout(node):
    nstyle = NodeStyle()
    nstyle["hz_line_width"] = 1  # Increase the width of the horizontal lines
    nstyle["vt_line_width"] = 1
    # If node is a leaf, add the node's name as a face object
    if node.is_leaf:
        # Create a NodeStyle for each leaf

        nstyle["size"] = 0  # Set the size of the node
        node.set_style(nstyle)

        # Creates a RectFace that will be drawn with the "aligned" option in
        color = get_color(node.name.split('_')[-1])
        color_face = RectFace(20 * scalar, 10 * scalar, color, color)  # Change the color as needed
        add_face_to_node(color_face, node, column=1, aligned=True)
    else:

        nstyle["size"] = 0  # Set the size of the node
        node.set_style(nstyle)


class TreeBuilder:
    family_map = defaultdict()

    def __init__(self, distmap: pd.DataFrame = None, tree_data: str = None, is_truth: bool = True, output_file_suffix: str = '') -> None:
        assert (distmap is None and tree_data is None), "Either a distmap or a tree_data must be provided"
        assert (distmap is not None or tree_data is not None), "Distmap  and tree_data are mutually exclusive"

        if distmap is not None:
            self.distmap = distmap.values
            self.names = list(distmap.columns)
            assert len(self.names) == distmap.shape[0] == distmap.shape[1]

        self.out_suffix = 'truth' if is_truth else 'pred' + output_file_suffix

    def get_tree(self) -> tuple[np.array, Tree]:
        """
        Computes the the tree from precomputed distances and saves the tree to disk. It also
        computes and returns the cophenetic distance matrix.
        Args:
            method: The tree building method to use.

        Returns:
            Cophentic distance matrix.
        """

        out_filepath = os.path.join(os.environ['OUTPUT_DIR'], f'%s_nj_{self.out_suffix}')
        tree_data = self.compute_tree()
        # store the newick file
        newick_filepath = out_filepath % 'Treedata' + '.nw'
        logger.debug(f'Writing newick to {newick_filepath}')
        with open(newick_filepath, 'w') as file:
            file.write(tree_data)
        tree = self.draw_tree(out_filepath, tree_data)

        return np.array(tree.cophenetic_matrix()[0]), tree

    def compute_tree(self) -> str:
        """
        Compute a tree on the distance matrix using the specified method.
        Args:
            method:

        Returns:
            The newick string of the tree.
        """
        # create a tree
        logger.debug(f'Constructing nj tree')
        # wirte distmap to local file
        named_tmp_file = self._write_distmap_phylib()
        # run rapidnj
        cmd = (f'/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/evaluation_visualization/rapidnj '
               f'{named_tmp_file.name} -i pd -o t -c 12')
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
        newick_data = p.stdout.read().decode('utf-8')
        newick_data = re.sub(r'.*% \r', '', newick_data)
        logger.debug("Finished tree construction")
        return newick_data

    def _write_distmap_phylib(self) -> tempfile.NamedTemporaryFile:
        sep = '\t'
        distmap_phylibfile = tempfile.NamedTemporaryFile(mode='w', suffix='.phylib')
        distmap_phylibfile.write(f'{len(self.names)}\n')
        for name, row in zip(self.names, self.distmap):
            values_str = sep.join(map(str, row))
            distmap_phylibfile.write(f'{name}\t{values_str}\n')
        distmap_phylibfile.flush()
        return distmap_phylibfile

    def draw_tree(self, out_filepath: str, tree_data: str) -> Tree:
        """
        Draw the tree using ete4 and save the tree to disk.
        Args:
            out_filepath: The path where to save it to
            tree_data: The tree in a newick format

        Returns:
            None
        """
        num_names = len(self.names)
        # draw the tree
        t = Tree(tree_data.format("newick"), parser=1)
        circular_style = TreeStyle()
        circular_style.show_leaf_name = False
        circular_style.mode = 'c'  # draw tree in circular mode
        circular_style.scale = num_names // 2
        circular_style.layout_fn = layout
        # Write the tree to output
        tree_vis_path = out_filepath % 'Tree' + '.png'
        logger.debug(f'Writing tree to {tree_vis_path}')
        t.render(tree_vis_path, w=1000, units='mm', tree_style=circular_style)

        return t
