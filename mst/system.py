import os
import torch
import auraloss
import pytorch_lightning as pl
from argparse import ArgumentParser

from typing import Callable


class System(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        generate_mix_console: torch.nn.Module,
        mix_fn: Callable,
        loss: torch.nn.Module,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.generate_mix_console = generate_mix_console
        self.mix_fn = mix_fn
        self.loss = loss

        # losses for evaluation
        self.sisdr = auraloss.time.SISDRLoss()
        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[512, 2048, 8192],
            hop_sizes=[256, 1024, 4096],
            win_lengths=[512, 2048, 8192],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )

    def forward(self, tracks: torch.Tensor, ref_mix: torch.Tensor) -> torch.Tensor:
        """Apply model to audio waveform tracks.
        Args:
            tracks (torch.Tensor): Set of input tracks with shape (bs, num_tracks, 1, seq_len)
            ref_mix (torch.Tensor): Reference mix with shape (bs, 2, seq_len)

        Returns:
            pred_mix (torch.Tensor): Predicted mix with shape (bs, 2, seq_len)
        """
        return self.model(tracks, ref_mix)

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
        tracks = batch

        # create a random mix (on GPU, if applicable)
        ref_mix, ref_param_dict = self.mix_fn(tracks, self.generate_mix_console)

        # now split into A and B sections
        middle_idx = ref_mix.shape[-1] // 2
        ref_mix_a = ref_mix[..., :middle_idx]
        ref_mix_b = ref_mix[..., middle_idx:]
        tracks_a = tracks[..., :middle_idx]
        tracks_b = tracks[..., middle_idx:]  # not used currently

        # process tracks from section A using reference mix from section B
        pred_mix_a, pred_param_dict = self(tracks_a, ref_mix_b)

        # compute loss on the predicted section A mix verus the ground truth reference mix
        loss = self.loss(pred_mix_a, ref_mix_a)

        # log the overall loss
        self.log(
            ("train" if train else "val") + "/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        sisdr_error = -self.sisdr(pred_mix_a, ref_mix_a)
        # log the SI-SDR error
        self.log(
            ("train" if train else "val") + "/si-sdr",
            sisdr_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        mrstft_error = self.mrstft(pred_mix_a, ref_mix_a)
        # log the MR-STFT error
        self.log(
            ("train" if train else "val") + "/mrstft",
            mrstft_error,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # for plotting down the line
        data_dict = {
            "ref_mix_a": ref_mix_a.detach().float().cpu(),
            "pred_mix_a": pred_mix_a.detach().float().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, data_dict = self.common_step(batch, batch_idx, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)

        if batch_idx == 0:
            return loss, data_dict

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
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
