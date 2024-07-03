from collections import defaultdict
from typing import Any, Dict
import seaborn as sns
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from torch import nn, optim

from build_dimreduction.utils.seeding import seed_worker
from build_dimreduction.datasets.collate_funcs import my_collate
from build_dimreduction.utils.triplet_mining import set_embedding_pairings
from build_dimreduction.datasets.triplet_sampling_dataset import TripletSamplingDataset


class FF_Triplets(pl.LightningModule):

    def __init__(self, dataset: TripletSamplingDataset, input_dim: int, hidden_dim: int, output_dim: int, lr: float, weight_decay: float,
                 postive_threshold: float, negative_threshold: float, non_linearity=nn.ReLU(), batch_size=32,leeway=1):
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
        self.dataset.set_constants(pos_threshold=postive_threshold, neg_threshold=negative_threshold,leeway=leeway)
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

        pos_cosine_dists = 1 - F.cosine_similarity(anchor_embeddings, positive_embeddings, dim=-1)
        neg_cosine_dists = 1 - F.cosine_similarity(anchor_embeddings, negative_embeddings, dim=-1)

        # Compute the triplet loss with margin
        loss = torch.relu(pos_cosine_dists - neg_cosine_dists + 0.3)

        set_embedding_pairings(self.dataset.prott5_embeddings, self.forward, self.device)

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
            ax = sns.heatmap(self.dataset.embedding_space_distances)
            ax.set_title(f'Distance Matrix Pred {self.current_epoch}')
            plt.show()
            self.dataset.polt_triplet_sampling(epoch=self.current_epoch, input_type='gt')
            self.dataset.polt_triplet_sampling(epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        # )
        return [optimizer]  # , [lr_scheduler]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=8, worker_init_fn=seed_worker, collate_fn=my_collate,
                                           sampler=None)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size, pin_memory=True,
                                           num_workers=8, worker_init_fn=seed_worker, collate_fn=my_collate,
                                           sampler=None, )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.dataset.serialize_for_storage()
    def on_train_epoch_start(self) -> None:
        self.dataset.set_shared_resources()


