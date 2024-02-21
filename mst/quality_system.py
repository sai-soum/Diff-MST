import os
import torch
import itertools
import pytorch_lightning as pl

from typing import Callable
from mst.utils import batch_stereo_peak_normalize

import warnings

warnings.filterwarnings(
    "ignore"
)  # fix this later to catch warnings about reading mp3 files


class QualityEstimationSystem(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        schedule: str = "step",
        lr: float = 3e-4,
        max_epochs: int = 500,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(encoder.embed_dim, 2 * encoder.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * encoder.embed_dim, 1),
        )
        self.save_hyperparameters(ignore=["encoder"])

    def forward(
        self,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        # could consider masking different parts of input and output
        # so the model cannot rely on perfectly aligned inputs

        z = self.encoder(mix)

        # project to parameter space
        pred = self.projector(z)

        return pred

    def common_step(
        self,
        batch: tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = False,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing rmix, stems and orig mix
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
            train (bool): Wether step is called during training (True) or validation (False).
        """
        mix, label = batch

        pred_label = self(mix).squeeze()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_label, label)

        # log the losses
        self.log(
            ("train" if train else "val") + "/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # compute accuracy
        pred_label = torch.sigmoid(pred_label)
        pred_label = torch.round(pred_label)
        acc = torch.sum(pred_label == label) / label.numel()
        self.log(
            ("train" if train else "val") + "/acc",
            acc,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.common_step(batch, batch_idx, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, train=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            itertools.chain(self.encoder.parameters(), self.projector.parameters()),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        if self.hparams.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs
            )
        elif self.hparams.schedule == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                [
                    int(self.hparams.max_epochs * 0.85),
                    int(self.hparams.max_epochs * 0.95),
                ],
            )
        else:
            return optimizer
        lr_schedulers = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], lr_schedulers
