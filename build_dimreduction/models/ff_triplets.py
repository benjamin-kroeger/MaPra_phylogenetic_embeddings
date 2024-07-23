from collections import defaultdict
from typing import Any, Dict

import numpy as np
import seaborn as sns
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler

from build_dimreduction.utils.seeding import seed_worker
from build_dimreduction.datasets.collate_funcs import my_collate
from build_dimreduction.utils.triplet_mining import set_per_dataname_pairings, set_embedding_pairings
from build_dimreduction.datasets.triplet_sampling_dataset import TripletSamplingDataset


class FF_Triplets(pl.LightningModule):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float, weight_decay: float,
                 postive_threshold: float, negative_threshold: float, non_linearity=nn.ReLU(), batch_size=32, leeway=1):
        super().__init__()
        self.save_hyperparameters()

        self.ff_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            non_linearity,
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            non_linearity,
            nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )
        self.batch_size = batch_size
        self.validation_outputs = defaultdict(list)

        # self.dataset = dataset
        # self.dataset.set_constants(pos_threshold=postive_threshold, neg_threshold=negative_threshold, leeway=leeway)
        # self.dataset.set_gt_pairings()
        # self.dataset.plot_distance_maps(distance_type='cophentic', mode='dist')
        #
        # # this part is hacky as fuck but i am out of time
        #
        # dataset_size = len(self.dataset)
        # indices = list(range(dataset_size))
        # split = int(np.floor(0.2 * dataset_size))
        # np.random.seed(42)
        # np.random.shuffle(indices)
        # train_indices, val_indices = indices[split:], indices[:split]
        #
        # self.train_sampler = SubsetRandomSampler(train_indices)
        # self.val_sampler = SubsetRandomSampler(val_indices)

    def forward(self, embeddings) -> Any:
        return self.ff_layer(embeddings)

    def computing_step(self, batch, batch_idx, mode: str = 'train'):
        anchor = batch[:, 0, :]
        positive = batch[:, 1, :]
        negative = batch[:, 2, :]

        anchor_embeddings = self.ff_layer(anchor)
        positive_embeddings = self.ff_layer(positive)
        negative_embeddings = self.ff_layer(negative)

        pos_cosine_dists = 1 - F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1)
        neg_cosine_dists = 1 - F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=-1)

        # Compute the triplet loss with margin
        loss = torch.relu(pos_cosine_dists - neg_cosine_dists + 1)

        # Average the loss over the batch
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx)
        self.log(name='train_loss', value=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx, 'val')
        self.validation_outputs['val_loss'].append(loss.cpu().item())

    def on_validation_epoch_end(self):
        # log average val_loss
        self.log('val_loss', torch.tensor(self.validation_outputs['val_loss']).mean())

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        return [optimizer]  # , [lr_scheduler]

