from collections import defaultdict
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, optim

from build_dimreduction.utils.seeding import seed_worker


class FF_Simple(pl.LightningModule):

    def __init__(self, dataset, input_dim: int, hidden_dim: int, output_dim: int, lr: float, weight_decay: float,
                 sampling_threshold: float, non_linearity=nn.ReLU(),batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.ff_layer = nn.Sequential(
            nn.Linear(in_features=self.hparams.input_dim, out_features=self.hparams.hidden_dim),
            self.hparams.non_linearity,
            nn.Linear(in_features=self.hparams.hidden_dim, out_features=self.hparams.output_dim)
        )
        self.batch_size = batch_size
        self.validation_outputs = defaultdict(list)
        self.dataset = dataset
        self.dataset.compute_sampling_pairs(threshold=sampling_threshold)
        assert self.dataset.sample_pairs is not None, "Sampling pair computation failed"

    def forward(self, embeddings) -> Any:
        return self.ff_layer(embeddings)

    def computing_step(self, batch, batch_idx):
        anchor = batch[:, 0, :]
        positive = batch[:, 1, :]

        anchor_embeddings = self.ff_layer(anchor)
        positive_embeddings = self.ff_layer(positive)

        #cosine_similarity = 1 - F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1)
        #loss = cosine_similarity.mean()

        #euclidean_distance = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=-1)
        #loss = euclidean_distance.mean()

        # Pairwise Distance
        pairwise_distance = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        loss = pairwise_distance.mean()

        return loss

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

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        return [optimizer]  # , [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=4, worker_init_fn=seed_worker,
                                           sampler=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=4, worker_init_fn=seed_worker,
                                           sampler=None)
