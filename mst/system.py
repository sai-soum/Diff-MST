import os
import json
import yaml
import torch
import auraloss
import pytorch_lightning as pl
from argparse import ArgumentParser

from typing import Callable
from mst.mixing import knowledge_engineering_mix
from mst.modules import causal_crop


class System(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        generate_mix_console: torch.nn.Module,
        mix_fn: Callable,
        loss: torch.nn.Module,
        use_track_loss: bool = False,
        use_mix_loss: bool = True,
        instrument_id_json: str = "data/instrument_name2id.json",
        knowledge_engineering_yaml: str = "data/knowledge_engineering.yaml",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.generate_mix_console = generate_mix_console
        self.mix_fn = mix_fn
        self.loss = loss
        self.use_track_loss = use_track_loss
        self.use_mix_loss = use_mix_loss

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

        # load configuration files
        if mix_fn is knowledge_engineering_mix:
            with open(instrument_id_json, "r") as f:
                self.instrument_number_lookup = json.load(f)

            with open(knowledge_engineering_yaml, "r") as f:
                self.knowledge_engineering_dict = yaml.safe_load(f)
        else:
            self.instrument_number_lookup = None
            self.knowledge_engineering_dict = None

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
        tracks, instrument_id, stereo_info = batch

        # create a random mix (on GPU, if applicable)
        (
            ref_mix_tracks,
            ref_mix,
            ref_track_param_dict,
            ref_fx_bus_param_dict,
            ref_master_bus_param_dict,
        ) = self.mix_fn(
            tracks,
            self.generate_mix_console,
            instrument_id,
            stereo_info,
            self.instrument_number_lookup,
            self.knowledge_engineering_dict,
        )

        if torch.isnan(ref_mix).any():
            print(ref_track_param_dict)
            raise ValueError("Found nan in ref_mix")

        # now split into A and B sections
        middle_idx = ref_mix.shape[-1] // 2

        ref_mix_a = ref_mix[..., :middle_idx]
        ref_mix_tracks_a = ref_mix_tracks[..., :middle_idx]
        ref_mix_b = ref_mix[..., middle_idx:]
        ref_mix_tracks_b = ref_mix_tracks[..., middle_idx:]
        tracks_a = tracks[..., :middle_idx]
        tracks_b = tracks[..., middle_idx:]  # not used currently

        bs, num_tracks, seq_len = tracks_a.shape

        # process tracks from section A using reference mix from section B
        (
            pred_mix_tracks_a,
            pred_mix_a,
            pred_track_param_dict,
            fx_bus_param_dict,
            ref_master_bus_param_dict,
        ) = self(tracks_a, ref_mix_b)

        # crop the target mix if it is longer than the predicted mix
        if ref_mix_a.shape[-1] > pred_mix_a.shape[-1]:
            seq_len = pred_mix_a.shape[-1]
            ref_mix_tracks_a = causal_crop(ref_mix_tracks_a, pred_mix_a.shape[-1])
            ref_mix_a = causal_crop(ref_mix_a, pred_mix_a.shape[-1])

        # compute loss on the predicted section A mix vs the ground truth reference mix
        loss = 0
        if self.use_mix_loss:
            mix_loss = self.loss(pred_mix_a, ref_mix_a)
            loss += mix_loss

        if self.use_track_loss:
            track_loss = self.loss(
                pred_mix_tracks_a.view(bs, num_tracks * 2, seq_len),
                ref_mix_tracks_a.view(bs, num_tracks * 2, seq_len),
            )
            loss += track_loss

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
            "ref_mix_b": ref_mix_b.detach().float().cpu(),
            "pred_mix_a": pred_mix_a.detach().float().cpu(),
            "sum_mix_a": tracks_a.sum(dim=1, keepdim=True).detach().float().cpu(),
        }

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, data_dict = self.common_step(batch, batch_idx, train=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)
        return data_dict

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
