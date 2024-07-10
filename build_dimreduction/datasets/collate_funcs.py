import torch


def my_collate(batch):
    """
    Custom collate function to deal with samples with different sizes.
    e.g: [4,3,1024], [5,3,1024]
    Torch fails to do this on its own
    Args:
        batch: A list of tensors.

    Returns:
        A concatenated batch of tensors.
    """
    return torch.cat(batch, dim=0)
