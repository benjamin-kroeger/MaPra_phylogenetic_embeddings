from collections import defaultdict
from typing import Any
import seaborn as sns
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import nn, optim

from build_dimreduction.utils.seeding import seed_worker
from build_dimreduction.datasets.collate_funcs import my_collate


class FF_Triplets(pl.LightningModule):

    def __init__(self, dataset, input_dim: int, hidden_dim: int, output_dim: int, lr: float, weight_decay: float,
                 postive_threshold: float, negative_threshold: float, non_linearity=nn.ReLU(), batch_size=32):
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

        self.dataset = dataset
        self.dataset.set_thresholds(pos_threshold=postive_threshold, neg_threshold=negative_threshold)
        self.dataset.set_gt_pairings()

    def forward(self, embeddings) -> Any:
        return self.ff_layer(embeddings)

    def computing_step(self, batch, batch_idx):
        anchor = batch[:, 0, :]
        positive = batch[:, 1, :]
        negative = batch[:, 2, :]

        anchor_embeddings = self.ff_layer(anchor)
        positive_embeddings = self.ff_layer(positive)
        negative_embeddings = self.ff_layer(negative)

        pos_cosine_dists = 1 - F.relu(F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1))
        neg_cosine_dists = 1 - F.relu(F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=-1))

        # Compute the triplet loss with margin
        loss = torch.relu(pos_cosine_dists - neg_cosine_dists + 0.3)

        # Average the loss over the batch
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx)
        self.log(name='train_loss', value=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx)
        self.validation_outputs['val_loss'].append(loss.cpu().item())

    def on_validation_epoch_end(self):
        # log average val_loss
        self.log('val_loss', torch.tensor(self.validation_outputs['val_loss']).mean())

        if self.current_epoch % 10 == 0:
            ax = sns.heatmap(self.dataset.compute_embedding_distances(self.forward))
            ax.set_title(f'Distance Matrix Pred {self.current_epoch}')
            plt.show()
            self.dataset.polt_triplet_sampling()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        return [optimizer]  # , [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=1, worker_init_fn=seed_worker, collate_fn=my_collate,
                                           sampler=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=1, worker_init_fn=seed_worker, collate_fn=my_collate,
                                           sampler=None, )
