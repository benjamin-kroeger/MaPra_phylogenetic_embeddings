import numpy as np
import os
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from evaluation_visualization.utils import get_color,color_assignment

def get_umap(distance_matrix) -> np.array:
    out_filepath = os.path.join(os.environ['OUTPUT_DIR'], f'Umap_.png')

    umap_calc = umap.UMAP(n_components=2, random_state=42, densmap=True, n_jobs=8,
                          metric='precomputed',
                          output_metric='euclidean')
    umap_embedding = umap_calc.fit_transform(distance_matrix)

    colors = [get_color(seq_name.split('_')[-1]) for seq_name in distance_matrix.columns]

    plt.figure(figsize=(10, 7))
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1],c=colors)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
                       for label, color in color_assignment.items()]
    plt.legend(handles=legend_elements, title='Labels')
    plt.title('UMAP projection')
    plt.savefig(out_filepath)

    # distances_umap = distance.cdist(umap_embedding.embedding_, umap_embedding.embedding_, metric='euclidean')
    # return distances_umap, umap_embedding

