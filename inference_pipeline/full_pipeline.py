import argparse
import os
from glob import glob
import torch
from matplotlib import pyplot as plt

from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from build_dimreduction.models.ff_simple import FF_Simple
from build_dimreduction.models.ff_triplets import FF_Triplets
from inference_pipeline.embedding_distance_metrics import sim_scorer
from evaluation_visualization.analysis_pipeline import analyse_distmaps
import pandas as pd
import numpy as np
import seaborn as sns


# 1. get prott5_embeddings
# 2. load model
# 3. compute dim reduction
# 4. compute distance matrix
# 5. get visualizations

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoint', type=str, required=False,default="newest", help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to the fasta file')

    args = parser.parse_args()

    return args


def _dim_reduction(embeddings, args):
    checkpoint_path = args.checkpoint
    if args.checkpoint == "newest":
        checkpoint_path = get_newest_file()
    model = FF_Triplets.load_from_checkpoint(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = embeddings.to(device)
    reduced = model(embeddings)

    return reduced


def get_input_data(path_to_input_folder: str) -> tuple[str, str]:
    assert os.path.isdir(path_to_input_folder), 'Path does not exist!'

    files = glob(os.path.join(path_to_input_folder, '*'))
    fasta_file = [x for x in files if x.endswith('.fasta')][0]
    distance_file = [x for x in files if x.endswith('.csv')][0]

    return fasta_file, distance_file


def get_newest_file():
    files = glob(
        os.path.join("/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/build_dimreduction/Data/chpts", '*'))

    # Filter out directories and get only files
    files = [f for f in files if os.path.isfile(f)]

    # Check if the list is not empty
    if not files:
        return None

    # Get the newest file
    newest_file = max(files, key=os.path.getmtime)

    return newest_file


def main(args):
    fasta_path, distance_path = get_input_data(args.input)

    embedder = ProtSeqEmbedder('prott5')
    embedd_data = zip(*embedder.get_raw_embeddings(fasta_path))
    ids, embeddings = list(embedd_data)

    reduced_embeddings = _dim_reduction(torch.stack(embeddings), args).cpu().detach().numpy()

    #distance_matrix = 1- np.abs(sim_scorer.cosine_similarity(reduced_embeddings, reduced_embeddings))
    distance_matrix = np.abs(sim_scorer.euclidean_distance(reduced_embeddings, reduced_embeddings))
    distance_truth = pd.read_csv(distance_path, index_col=0)

    ax = sns.heatmap(distance_matrix)
    ax.set_title('Distance Matrix Pred')
    plt.show()
    ax = sns.heatmap(distance_truth)
    ax.set_title('Distance matrix Truth')
    plt.show()

    analyse_distmaps(distmap1_pred=pd.DataFrame(distance_matrix, index=ids, columns=ids), distmap2_truth=distance_truth)


if __name__ == '__main__':
    input_args = init_parser()
    main(input_args)
