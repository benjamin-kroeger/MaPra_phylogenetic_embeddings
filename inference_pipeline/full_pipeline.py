import argparse
import os
from glob import glob
import torch
from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from build_dimreduction.models.ff_simple import FF_Simple
from inference_pipeline.embedding_distance_metrics import sim_scorer
from evaluation_visualization.analysis_pipeline import analyse_distmaps
import pandas as pd

# 1. get embeddings
# 2. load model
# 3. compute dim reduction
# 4. compute distance matrix
# 5. get visualizations

def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Path to the fasta file')

    args = parser.parse_args()

    return args


def _dim_reduction(embeddings, args):
    model = FF_Simple.load_from_checkpoint(args.checkpoint)

    reduced = model(embeddings)

    return reduced


def get_input_data(path_to_input_folder: str) -> tuple[str, str]:
    assert os.path.isdir(path_to_input_folder), 'Path does not exist!'

    files = glob(os.path.join(path_to_input_folder, '*'))
    fasta_file = [x for x in files if x.endswith('.fasta')][0]
    distance_file = [x for x in files if x.endswith('.csv')][0]

    return fasta_file, distance_file


def main(args):
    fasta_path, distance_path = get_input_data(args.input)

    embedder = ProtSeqEmbedder('prott5')
    embedd_data = zip(*embedder.get_raw_embeddings(fasta_path))
    ids, embeddings = list(embedd_data)

    reduced_embeddings = _dim_reduction(torch.stack(embeddings), args).cpu().detach().numpy()

    distance_matrix = sim_scorer.euclidean_distance(reduced_embeddings, reduced_embeddings)

    analyse_distmaps(distmap1_pred=pd.DataFrame(distance_matrix, index=ids, columns=ids), distmap2_truth=pd.read_csv(distance_path, index_col=0))


if __name__ == '__main__':
    input_args = init_parser()
    main(input_args)
