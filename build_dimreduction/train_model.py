import argparse
from datetime import datetime

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GradientAccumulationScheduler, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from build_dimreduction.datasets.collate_funcs import my_collate
from build_dimreduction.datasets.triplet_sampling_dataset import TripletSamplingDataset, UpdateDatasetCallback
from build_dimreduction.utils.triplet_mining import set_embedding_pairings
from models.ff_triplets import FF_Triplets


def init_parser():
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Protein Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Performance related arguments
    parser.add_argument('--input_folder', type=str, nargs='+', required=True, help='Path to input folder')
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='CPU Cores')
    parser.add_argument('--half_precision', action='store_true', default=False, help='Train the model with torch.float16 instead of torch.float32')
    parser.add_argument('--cpu_only', action='store_true', default=False, help='If the GPU is to WEAK use cpu only')
    parser.add_argument('--model', type=str, required=False, default='ff_triplets')

    parser.add_argument('--lr', type=float, required=False, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-3, help='Weight decay')
    parser.add_argument('--acc_grad', type=bool, default=False)
    parser.add_argument('--epochs', type=int, required=False, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=False, default=128, help='The batch_size')
    parser.add_argument('--hidden_dim', type=int, required=False, default=128, help='The size of the hidden layer')
    parser.add_argument('--output_dim', type=int, required=False, default=128, help='The size of the output layer')
    parser.add_argument('--positive_threshold', type=float, required=False, default=0.4, help='The threshold for whats considered a positive')
    parser.add_argument('--negative_threshold', type=float, required=False, default=0.7, help='The threshold for whats considered a negative')
    parser.add_argument('--leeway', type=int, required=False, default=1, help='How lax the sampling shall be')

    args = parser.parse_args()

    return args


def _setup_callback(args,model,dataset,device):
    """
    Sets up callbacks for training
    Args:
        args:

    Returns:
         A list of callbacks for training
    """
    # set up early stopping and storage of the best model
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.005, patience=20, verbose=False, mode="min")
    # set up storage of the best checkpoint path
    best_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode="min", dirpath="build_dimreduction/Data/chpts",
                                      filename=args.model + "_{epoch:02d}_{val_loss:.4f}", auto_insert_metric_name=True)
    # set up lr monitor for future
    lr_monitor = LearningRateMonitor(logging_interval='step')
    # update dataset callback
    update_dataset_callback = UpdateDatasetCallback(model=model,dataset=dataset,device=device)

    callbacks = [early_stop_callback, lr_monitor, best_checkpoint,update_dataset_callback]

    if args.acc_grad:
        # if needed gradient accumulation can be used
        accumulator = GradientAccumulationScheduler(scheduling={0: args.acc_grad})
        callbacks.append(accumulator)

    return callbacks


def get_model(args, device):
    """
    Set up the model and send it to the device
    Args:
        args:
        device:

    Returns:
        The model
    """
    # init model
    model = FF_Triplets(input_dim=1024, hidden_dim=args.hidden_dim, output_dim=args.output_dim, lr=args.lr,
                        weight_decay=args.weight_decay,
                        postive_threshold=args.positive_threshold,
                        negative_threshold=args.negative_threshold,
                        batch_size=args.batch_size)
    model.to(device)
    return model


def get_dataset(args, device, forward):
    """
    Initalize the dataset and set constants, pairings and plot inital state
    Args:
        args: the input args to set up the dataset
        device: The device on which the dataset shall sit
        forward: The models forward function to create embeddings for embedd distances

    Returns:
        The dataset
    """
    # init dataset
    dataset = TripletSamplingDataset('prott5', args.input_folder, device=device)
    # set the thresholds for ground truths
    dataset.set_constants(pos_threshold=args.positive_threshold, neg_threshold=args.negative_threshold, leeway=args.leeway)
    # plot the distance distributions
    dataset.plot_distance_maps(distance_type='cophentic', mode='dist')
    # compute ground truth embeddings
    dataset.set_gt_pairings()
    # compute the embedding distances
    set_embedding_pairings(dataset, forward, device)

    return dataset


def main(args):
    # set the default dtype to float32
    dtype = torch.float32
    if args.half_precision:
        dtype = torch.float16

    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu_only:
        device = torch.device('cpu')

    # create an experiment name
    experiment_name = f'{args.model}-{datetime.now().strftime("%d%m%y_%H_%M")}'

    # initialize 5 fold cross validation
    for fold in range(1):
        model = get_model(args, device)
        dataset = get_dataset(args, device, model.forward)

        callbacks = _setup_callback(args,model, dataset, device)
        # set up a logger
        wandb_logger = WandbLogger(name=f'{experiment_name}_{fold}', project='MaPra')
        wandb_logger.watch(model)
        # add experiment name so that we can group runs in wandb
        wandb_logger.experiment.config['experiment'] = experiment_name

        trainer = pl.Trainer(
            # precision='16-mixed' if args.half_precision else 32,
            max_epochs=args.epochs,
            accelerator='gpu' if device == torch.device('cuda') else 'cpu',
            devices=1,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            logger=wandb_logger,
            log_every_n_steps=3
        )
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
                                                       num_workers=8, collate_fn=my_collate,shuffle=True,
                                                       )
        val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, pin_memory=True,
                                                       num_workers=8, collate_fn=my_collate,
                                                       )
        # train the model
        trainer.fit(model, train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)
        # load the best model and run one final validation
        best_model_path = trainer.checkpoint_callback.best_model_path

        wandb_logger.finalize('success')
        wandb.finish()

        return best_model_path


if __name__ == '__main__':
    pl.seed_everything(42)
    cmd_args = init_parser()
    main(cmd_args)
