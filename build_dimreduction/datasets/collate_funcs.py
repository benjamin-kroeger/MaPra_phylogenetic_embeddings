import torch


def my_collate(batch):
    return torch.cat(batch, dim=0)
