import os
import csv
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger


from mst.systems import MixStyleTransfer
from mst.callbacks.audio import LogAudioCallback
from mst.dataloaders.musdb import MUSEDB_on_the_fly, mix_gen
from mst.dataloaders.medley import MedleyDB_on_fly




if __name__ == "__main__":
    print(torch.cuda.is_available())
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42, workers=True)
    parser = ArgumentParser()

    # add PROGRAM level args

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--loss_criteria", type=str, default="STFTloss")
    parser.add_argument("--dataset", type=str, default="MUSEDB")
    parser.add_argument("--model", type=str, default="encoder_with_resnet")
    parser.add_argument("--experiment_type", type=str, default="gain_pan")
    parser.add_argument("--num_tracks", type=int, default=1)
    parser.add_argument("--diff_sections", type=bool, default=True)
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=5) 
    parser.add_argument("--log_dir", type=str, default="./logs")

    parser = MixStyleTransfer.add_model_specific_args(parser)  # add model specific args
    parser = pl.Trainer.add_argparse_args(parser)  # add all Trainer options
    args = parser.parse_args()  # parse them args

    # setup callbacks
    callbacks = [
        LogAudioCallback(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            filename=f"{args.dataset}-{args.model}-{args.experiment_type}" + "_epoch-{epoch}-step-{step}",
            monitor="val/loss_epoch",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]

    wandb_logger = WandbLogger(save_dir=args.log_dir, project="mst")

    # create PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=callbacks
    )

    # create the System
    system = MixStyleTransfer(**vars(args))

    if "MUSEDB" in args.dataset:
        train_dataset = MUSEDB_on_the_fly(
            type = "train", 
            experiment_type = args.experiment_type, 
            num_tracks = args.num_tracks, 
            diff_sections = args.diff_sections, 
            duration = args.duration)
        val_dataset = MUSEDB_on_the_fly(
            type = "val",   
            experiment_type = args.experiment_type,
            num_tracks = args.num_tracks,
            diff_sections = args.diff_sections,
            duration = args.duration)
        test_dataset = MUSEDB_on_the_fly(   
            type = "test",
            experiment_type = args.experiment_type,
            num_tracks = args.num_tracks,
            diff_sections = args.diff_sections,
            duration = args.duration)
        
#implement indices and trainset, valset, testset
    elif args.dataset == "MedleyDB":
        train_dataset = MedleyDB_on_fly(
            type = "train", 
            experiment_type = args.experiment_type, 
            num_tracks = args.num_tracks, 
            diff_sections = args.diff_sections, 
            duration = args.duration,
            indices = [0,100])
        val_dataset =MedleyDB_on_fly(
            type = "val",   
            experiment_type = args.experiment_type,
            num_tracks = args.num_tracks,
            diff_sections = args.diff_sections,
            duration = args.duration,
            indices = [100,115])
        test_dataset = MedleyDB_on_fly(   
            type = "test",
            experiment_type = args.experiment_type,
            num_tracks = args.num_tracks,
            diff_sections = args.diff_sections,
            duration = args.duration,
            indices = [115, 140])
        train_set = train_dataset.mix_dirs
        val_set = val_dataset.mix_dirs
        test_set = test_dataset.mix_dirs



    # create a file with the train/val/test split
    csv_filepath = os.path.join(wandb_logger.experiment.dir, "split.csv")
    print(csv_filepath)
    with open(csv_filepath, "w") as f:
        writer = csv.writer(f)
        for fp in train_set:
            writer.writerow(["train", fp])
        for fp in val_set:
            writer.writerow(["val", fp])
        for fp in test_set:
            writer.writerow(["test", fp])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        #pin_memory=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        #pin_memory=True, # does this help or hurt? 
    )

    # train!
    trainer.fit(system, train_dataloader, val_dataloader)