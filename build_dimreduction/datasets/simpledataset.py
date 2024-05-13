import torch
from torch.utils.data import Dataset
from build_dimreduction.utils.get_raw_embeddings import ProtSeqEmbedder


class SimpleDataset(Dataset):
    def __init__(self, model: str):
        embedder = ProtSeqEmbedder(model)
        embedd_data = zip(*embedder.get_raw_embeddings(
            '/home/benjaminkroeger/Documents/Master/Master_3_Semester/MaPra/Learning_phy_distances/input_data/input_case/test1/phosphatase.fasta'))
        self.ids, self.embeddings = list(embedd_data)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Calculate adjusted index for the second embedding
        adjusted_idx = (idx + 1) % len(self.embeddings)
        return torch.cat([self.embeddings[idx][None, :], self.embeddings[adjusted_idx][None, :]], dim=0)

