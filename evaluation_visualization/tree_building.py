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

# from umap import plot

logging.config.fileConfig(
    '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/logging.config',
    disable_existing_loggers=False)
logger = logging.getLogger(__name__)

color_palette = sns.color_palette("colorblind", 20)
color_assignment = {}


def get_color(familiy_name: str) -> str:
    if familiy_name in color_assignment.keys():
        return color_assignment[familiy_name]

    new_color = color_palette[len(color_assignment)]
    color_assignment[familiy_name] = '#{:02x}{:02x}{:02x}'.format(int(new_color[0]*255), int(new_color[1]*255), int(new_color[2]*255))

    return color_assignment[familiy_name]


def _square_to_lower_triangle(dist_square):
    n = len(dist_square)
    dist_lower_triangle = []
    for i in range(n):
        row = []
        for j in range(i + 1):
            row.append(dist_square[i][j])
        dist_lower_triangle.append(row)
    return dist_lower_triangle


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
        color_face = RectFace(20, 10, color, color)  # Change the color as needed
        add_face_to_node(color_face, node, column=1, aligned=True)
    else:

        nstyle["size"] = 0  # Set the size of the node
        node.set_style(nstyle)


class TreeBuilder:
    family_map = defaultdict()

    def __init__(self, distmap: pd.DataFrame, is_truth: bool, output_file_suffix: str = '') -> None:
        self.distmap = distmap.values
        self.out_suffix = 'truth' if is_truth else 'pred' + output_file_suffix
        self.names = list(distmap.columns)

        assert len(self.names) == distmap.shape[0] == distmap.shape[1]

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
        tree = self._draw_tree(out_filepath, tree_data)

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

    def _draw_tree(self, out_filepath: str, tree_data: str) -> Tree:
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
