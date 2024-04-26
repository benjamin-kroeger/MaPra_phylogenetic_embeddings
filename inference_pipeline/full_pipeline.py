import argparse
import torch
from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder
from build_dimreduction.models.ff_simple import FF_Simple
from embedding_distance_metrics import sim_scorer
from evaluation_visualization.analysis_pipeline import analyse_distmaps


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

    parser.add_argument('--checkpoint', type=str,required=True, help='Path to the model checkpoint')
    parser.add_argument('--fasta', type=str, required=True, help='Path to the fasta file')

    args = parser.parse_args()

    return args


def _dim_reduction(embeddings, args):
    model = FF_Simple.load_from_checkpoint(args.checkpoint)

    reduced = model(embeddings)

    return reduced


def main(args):
    embedder = ProtSeqEmbedder('prott5')
    embedd_data = zip(*embedder.get_raw_embeddings(args.fasta))
    ids, embeddings = list(embedd_data)

    reduced_embeddings = _dim_reduction(torch.stack(embeddings), args).cpu().detach().numpy()

    distance_matrix = sim_scorer.euclidean_distance(reduced_embeddings, reduced_embeddings)

    analyse_distmaps(distmap1=distance_matrix, distmap2=distance_matrix,names=list(ids))


if __name__ == '__main__':
    input_args = init_parser()
    main(input_args)
