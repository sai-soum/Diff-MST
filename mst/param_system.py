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


class ParameterEstimationSystem(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module,
        mix_console: torch.nn.Module,
        remixer: torch.nn.Module,
        schedule: str = "step",
        lr: float = 3e-4,
        max_epochs: int = 500,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mix_console = mix_console
        self.remixer = remixer
        self.projector = projector
        self.save_hyperparameters(
            ignore=["encoder", "mix_console", "remixer", "projector"]
        )

    def forward(
        self,
        input_mix: torch.Tensor,
        output_mix: torch.Tensor,
    ) -> torch.Tensor:
        z_in = self.encoder(input_mix)
        z_out = self.encoder(output_mix)

        # take difference between embeddings
        z_diff = z_out - z_in

        # project to parameter space
        track_params, fx_bus_params, master_bus_params = self.projector(z_diff)

        return track_params, fx_bus_params, master_bus_params

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
        input_mix = batch

        # create a remix
        output_mix, track_params, fx_bus_params, master_bus_params = self.remixer(
            input_mix, self.mix_console
        )

        # estimate parameters
        track_params_hat, fx_bus_params_hat, master_bus_params_hat = self(
            input_mix, output_mix
        )

        # calculate loss
        track_params_loss = torch.nn.functional.mse_loss(
            track_params_hat,
            track_params,
        )
        fx_bus_params_loss = torch.nn.functional.mse_loss(
            fx_bus_params_hat,
            fx_bus_params,
        )
        master_bus_params_loss = torch.nn.functional.mse_loss(
            master_bus_params_hat,
            master_bus_params,
        )

        loss = track_params_loss + fx_bus_params_loss + master_bus_params_loss

        # log the losses
        self.log(
            ("train" if train else "val") + "/track_param_loss",
            track_params_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            ("train" if train else "val") + "/fx_bus_param_loss",
            fx_bus_params_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            ("train" if train else "val") + "/master_bus_param_loss",
            master_bus_params_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

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
