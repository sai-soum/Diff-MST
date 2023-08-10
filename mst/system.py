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
        active_eq_step: int = 0,
        active_compressor_step: int = 0,
        active_fx_bus_step: int = 0,
        active_master_bus_step: int = 0,
        warmup: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.generate_mix_console = generate_mix_console
        self.mix_fn = mix_fn
        self.loss = loss
        self.use_track_loss = use_track_loss
        self.use_mix_loss = use_mix_loss
        self.active_eq_step = active_eq_step
        self.active_compressor_step = active_compressor_step
        self.active_fx_bus_step = active_fx_bus_step
        self.active_master_bus_step = active_master_bus_step
        self.warmup = warmup

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

        # disable parts of the mix console based on global step
        use_track_eq = True if self.global_step >= self.active_eq_step else False
        use_track_compressor = (
            True if self.global_step >= self.active_compressor_step else False
        )
        use_fx_bus = True if self.global_step >= self.active_fx_bus_step else False
        use_master_bus = (
            True if self.global_step >= self.active_master_bus_step else False
        )

        # --------- create a random mix (on GPU, if applicable) ---------
        (
            ref_mix_tracks,
            ref_mix,
            ref_track_param_dict,
            ref_fx_bus_param_dict,
            ref_master_bus_param_dict,
        ) = self.mix_fn(
            tracks,
            self.generate_mix_console,
            use_track_gain=True,
            use_track_panner=True,
            use_track_eq=use_track_eq,
            use_track_compressor=use_track_compressor,
            use_fx_bus=use_fx_bus,
            use_master_bus=use_master_bus,
            instrument_id=instrument_id,
            stereo_id=stereo_info,
            instrument_number_file=self.instrument_number_lookup,
            ke_dict=self.knowledge_engineering_dict,
            warmup=self.warmup,
        )

        if torch.isnan(ref_mix).any():
            print(ref_track_param_dict)
            raise ValueError("Found nan in ref_mix")

        # now split into A and B sections (accounting for warmup)
        ref_middle_idx = (tracks.shape[-1] // 2) - self.warmup
        input_middle_idx = tracks.shape[-1] // 2

        ref_mix_a = ref_mix[..., :ref_middle_idx]
        ref_mix_tracks_a = ref_mix_tracks[..., :ref_middle_idx]

        ref_mix_b = ref_mix[..., ref_middle_idx:]
        ref_mix_tracks_b = ref_mix_tracks[..., ref_middle_idx:]

        tracks_a = tracks[..., :input_middle_idx]
        tracks_b = tracks[..., input_middle_idx:]  # not used currently

        bs, num_tracks, seq_len = tracks_a.shape

        #  ---- run model with tracks from section A using reference mix from section B ----
        (
            pred_mix_tracks_b,
            pred_mix_b,
            pred_track_param_dict,
            pred_fx_bus_param_dict,
            pred_master_bus_param_dict,
        ) = self.model(
            tracks_b,
            ref_mix_a,
            use_track_gain=True,
            use_track_panner=True,
            use_track_eq=use_track_eq,
            use_track_compressor=use_track_compressor,
            use_fx_bus=use_fx_bus,
            use_master_bus=use_master_bus,
        )

        # don't compute error on start of the audio (warmup)
        ref_mix_b = ref_mix_b[..., self.warmup :]
        pred_mix_b = pred_mix_b[..., self.warmup :]

        # peak normalize mixes
        gain_lin = ref_mix_b.abs().max(dim=-1, keepdim=True)[0]
        gain_lin = gain_lin.max(dim=-2, keepdim=True)[0]
        ref_mix_b_norm = ref_mix_b / gain_lin.clamp(min=1e-8)

        gain_lin = pred_mix_b.abs().max(dim=-1, keepdim=True)[0]
        gain_lin = gain_lin.max(dim=-2, keepdim=True)[0]
        pred_mix_b_norm = pred_mix_b / gain_lin.clamp(min=1e-8)

        # compute loss on the predicted section A mix vs the ground truth reference mix
        loss = 0
        if self.use_mix_loss:
            mix_loss = self.loss(pred_mix_b_norm, ref_mix_b_norm)
            loss += mix_loss

        if self.use_track_loss:
            track_loss = self.loss(
                pred_mix_tracks_b.view(bs, num_tracks * 2, seq_len),
                ref_mix_tracks_b.view(bs, num_tracks * 2, seq_len),
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

        sisdr_error = -self.sisdr(pred_mix_b_norm, ref_mix_b_norm)
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

        mrstft_error = self.mrstft(pred_mix_b_norm, ref_mix_b_norm)
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
            "ref_mix_b_norm": ref_mix_b_norm.detach().float().cpu(),
            "pred_mix_b_norm": pred_mix_b_norm.detach().float().cpu(),
            "sum_mix_a": tracks_a.sum(dim=1, keepdim=True).detach().float().cpu(),
            "ref_track_param_dict": ref_track_param_dict,
            "pred_track_param_dict": pred_track_param_dict,
            "ref_fx_bus_param_dict": ref_fx_bus_param_dict,
            "pred_fx_bus_param_dict": pred_fx_bus_param_dict,
            "ref_master_bus_param_dict": ref_master_bus_param_dict,
            "pred_master_bus_param_dict": pred_master_bus_param_dict,
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
