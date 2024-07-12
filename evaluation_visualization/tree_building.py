import logging.config
import os
import re
import tempfile
from collections import defaultdict
from subprocess import PIPE, Popen, STDOUT

import numpy as np
import pandas as pd
# import umap.umap_ as umap
from ete4 import Tree
from ete4.treeview import TreeStyle, NodeStyle, RectFace, add_face_to_node

from evaluation_visualization.utils import get_color, convert_to_full

# from umap import plot

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)
# TODO set scalar based on longest branch
scalar = 1


def layout(node):
    nstyle = NodeStyle()
    nstyle["hz_line_width"] = int(1*scalar)  # Increase the width of the horizontal lines
    nstyle["vt_line_width"] = int(1*scalar)
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
    """
    This class provides utilities to plot a tree and compute sycophantic distances, either with a df or newick file as input
    """
    def __init__(self, distmap: pd.DataFrame = None, path_to_tree_data: str = None, is_truth: bool = True, output_file_suffix: str = '') -> None:
        assert (distmap is None) != (path_to_tree_data is None), "Either distmap or path_to_tree_data must be provided, but not both."

        # incase a distance matrix was provided
        if distmap is not None:
            self.distmap = distmap.values
            self.names = list(distmap.columns)
            assert len(self.names) == distmap.shape[0] == distmap.shape[1]
            self.tree_data = self._compute_tree()

        self.out_suffix = 'truth' if is_truth else 'pred' + output_file_suffix

        # workflow if the tree is already given
        if path_to_tree_data is not None:
            assert os.path.isfile(path_to_tree_data), "The path to tree_data must be an existing file"
            with open(path_to_tree_data, 'r') as newick_data:
                self.tree_data: str = newick_data.read()

        # convert newick tree to ete4 format
        self.ete_tree = Tree(self.tree_data, parser=1)
        # find longest


    def find_longest_path_iterative(self):

        max_length = 0
        stack = [(self.ete_tree, 0)]  # Each element in stack is a tuple (node, cumulative_distance)

        while stack:
            node, current_length = stack.pop()
            if node.is_leaf:
                max_length = max(max_length, current_length + node.dist)
            else:
                for child in node.children:
                    stack.append((child, current_length + child.dist))
        global scalar
        scalar = max(1,max_length // 3)

    def get_cophentic_distances(self) -> tuple[np.array, list[str]]:
        """
        Compute the cophentic distances on the tree
        Returns:
            Array of cophentic distances, names is same order
        """

        cophentic_distances, names = self.ete_tree.cophenetic_matrix()
        cophentic_distances = np.array(cophentic_distances)
        return cophentic_distances, names

    def _compute_tree(self) -> str:
        """
        Compute a tree on the distance matrix using the specified method.

        Returns:
            The newick string of the tree.
        """
        assert self.distmap is not None, "The provided distmap is invalid"
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

    def draw_tree(self, out_filepath: str) -> Tree:
        """
        Draw the tree using ete4 and save the tree to disk.
        Args:
            out_filepath: The path where to save it to
            tree_data: The tree in a newick format

        Returns:
            None
        """
        #self.find_longest_path_iterative() # not sure if this is even necessary
        # draw the tree
        circular_style = TreeStyle()
        circular_style.show_leaf_name = False
        circular_style.mode = 'c'  # draw tree in circular mode
        circular_style.layout_fn = layout
        # Write the tree to output
        tree_vis_path = out_filepath % 'Tree' + '.png'
        logger.debug(f'Writing tree to {tree_vis_path}')
        self.ete_tree.render(tree_vis_path, w=1000, units='mm', tree_style=circular_style)


if __name__ == '__main__':
    test_df = pd.read_csv(
        '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/input_data/input_case/kinase/kinase_distances.csv',
        index_col=0)
    test_df = convert_to_full(test_df)
    test = TreeBuilder(
        #path_to_tree_data='/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/output_dir/Treedata_nj_pred.nw',
        distmap=test_df
    )
    dists, names = test.get_cophentic_distances()
    test.draw_tree('/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/output_dir/%s')
