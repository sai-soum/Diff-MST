import os
import json
import yaml
import torch
import auraloss
import pytorch_lightning as pl

import time
from typing import Callable
from mst.mixing import knowledge_engineering_mix
from mst.utils import batch_stereo_peak_normalize
from mst.fx_encoder import FXencoder
import pyloudnorm as pyln


class System(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        mix_console: torch.nn.Module,
        mix_fn: Callable,
        loss: torch.nn.Module,
        generate_mix: bool = True,
        use_track_loss: bool = False,
        use_mix_loss: bool = True,
        use_param_loss: bool = False,
        instrument_id_json: str = "data/instrument_name2id.json",
        knowledge_engineering_yaml: str = "data/knowledge_engineering.yaml",
        active_eq_epoch: int = 0,
        active_compressor_epoch: int = 0,
        active_fx_bus_epoch: int = 0,
        active_master_bus_epoch: int = 0,
        lr: float = 1e-4,
        max_epochs: int = 500,
        schedule: str = "step",
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.mix_console = mix_console
        self.mix_fn = mix_fn
        self.loss = loss
        self.generate_mix = generate_mix
        self.use_track_loss = use_track_loss
        self.use_mix_loss = use_mix_loss
        self.use_param_loss = use_param_loss
        self.active_eq_epoch = active_eq_epoch
        self.active_compressor_epoch = active_compressor_epoch
        self.active_fx_bus_epoch = active_fx_bus_epoch
        self.active_master_bus_epoch = active_master_bus_epoch

        self.meter = pyln.Meter(44100)
        #self.warmup = warmup


        self.save_hyperparameters(ignore=["model", "mix_console", "mix_fn", "loss"])


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

        # default
        self.use_track_input_fader = True
        self.use_track_panner = True
        self.use_track_eq = False
        self.use_track_compressor = False
        self.use_fx_bus = False
        self.use_master_bus = False
        self.use_output_fader = True

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
        train: bool = False,
    ):
        """Model step used for validation and training.
        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing rmix, stems and orig mix
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
            train (bool): Wether step is called during training (True) or validation (False).
        """

        tracks, instrument_id, stereo_info, track_padding, ref_mix, song_name = batch
        #print("song_names from this batch: ", song_name)

        # split into A and B sections
        middle_idx = tracks.shape[-1] // 2

        # disable parts of the mix console based on global step
        if self.current_epoch >= self.active_eq_epoch:
            self.use_track_eq = True

        if self.current_epoch >= self.active_compressor_epoch:
            self.use_track_compressor = True

        if self.current_epoch >= self.active_fx_bus_epoch:
            self.use_fx_bus = True

        if self.current_epoch >= self.active_master_bus_epoch:
            self.use_master_bus = True

        bs, num_tracks, seq_len = tracks.shape

        # apply random gain to input tracks
        # tracks *= 10 ** ((torch.rand(bs, num_tracks, 1).type_as(tracks) * -12.0) / 20.0)
        ref_track_param_dict = None
        ref_fx_bus_param_dict = None
        ref_master_bus_param_dict = None

        # if tracks[...,middle_idx:].sum() == 0:
        #     print("tracks are zero")
        #     print(tracks[...,middle_idx:])
        #     raise ValueError("input tracks are zero")
            
        # --------- create a random mix (on GPU, if applicable) ---------
        if self.generate_mix:
            (
                ref_mix_tracks,
                ref_mix,
                ref_track_param_dict,
                ref_fx_bus_param_dict,
                ref_master_bus_param_dict,
                ref_mix_params, 
                ref_fx_bus_params, 
                ref_master_bus_params
            ) = self.mix_fn(
                tracks,
                self.mix_console,
                use_track_input_fader=False,  # do not use track input fader for training
                use_track_panner=self.use_track_panner,
                use_track_eq=self.use_track_eq,
                use_track_compressor=self.use_track_compressor,
                use_fx_bus=self.use_fx_bus,
                use_master_bus=self.use_master_bus,
                use_output_fader=False,  # not used because we normalize output mixes
                instrument_id=instrument_id,
                stereo_id=stereo_info,
                instrument_number_file=self.instrument_number_lookup,
                ke_dict=self.knowledge_engineering_dict,
            )

            # normalize the reference mix
            ref_mix = batch_stereo_peak_normalize(ref_mix)

            if torch.isnan(ref_mix).any():
                #print(ref_track_param_dict)
                raise ValueError("Found nan in ref_mix")
            
            
            # if torch.count_nonzero(ref_mix[...,0:middle_idx])< 1:
            #     print("ref_mix is zero")
            #     raise ValueError("ref_mix is zero")

            ref_mix_a = ref_mix[..., :middle_idx]  # this is passed to the model
            ref_mix_b = ref_mix[..., middle_idx:]  # this is used for loss computation

        else:
            # when using a real mix, pass the same mix to model and loss
            ref_mix_a = ref_mix
            ref_mix_b = ref_mix
        
        


        # tracks_a = tracks[..., :input_middle_idx] # not used currently
       
        #print("input tracks: ", tracks[...,middle_idx:])
        #print("ref_mix: ", ref_mix_a)


        if self.current_epoch >= self.active_compressor_epoch:
            self.use_track_compressor = True

        if self.current_epoch >= self.active_fx_bus_epoch:
            self.use_fx_bus = True

        if self.current_epoch >= self.active_master_bus_epoch:
            self.use_master_bus = True

        bs, num_tracks, seq_len = tracks.shape

        # apply random gain to input tracks
        # tracks *= 10 ** ((torch.rand(bs, num_tracks, 1).type_as(tracks) * -12.0) / 20.0)
        ref_track_param_dict = None
        ref_fx_bus_param_dict = None
        ref_master_bus_param_dict = None

        # --------- create a random mix (on GPU, if applicable) ---------
        if self.generate_mix:
            (
                ref_mix_tracks,
                ref_mix,
                ref_track_param_dict,
                ref_fx_bus_param_dict,
                ref_master_bus_param_dict,
                ref_mix_params,
                ref_fx_bus_params,
                ref_master_bus_params,
            ) = self.mix_fn(
                tracks,
                self.mix_console,
                use_track_input_fader=False,  # do not use track input fader for training
                use_track_panner=self.use_track_panner,
                use_track_eq=self.use_track_eq,
                use_track_compressor=self.use_track_compressor,
                use_fx_bus=self.use_fx_bus,
                use_master_bus=self.use_master_bus,
                use_output_fader=False,  # not used because we normalize output mixes
                instrument_id=instrument_id,
                stereo_id=stereo_info,
                instrument_number_file=self.instrument_number_lookup,
                ke_dict=self.knowledge_engineering_dict,
            )

            # normalize the reference mix
            ref_mix = batch_stereo_peak_normalize(ref_mix)

            if torch.isnan(ref_mix).any():
                print(ref_track_param_dict)
                raise ValueError("Found nan in ref_mix")

            ref_mix_a = ref_mix[..., :middle_idx]  # this is passed to the model
            ref_mix_b = ref_mix[..., middle_idx:]  # this is used for loss computation
            # tracks_a = tracks[..., :input_middle_idx] # not used currently
            tracks_b = tracks[..., middle_idx:]  # this is passed to the model
        else:
            # when using a real mix, pass the same mix to model and loss
            ref_mix_a = ref_mix
            ref_mix_b = ref_mix
            tracks_b = tracks

       
        #  ---- run model with tracks from section A using reference mix from section B ----
        (
            pred_track_params,
            pred_fx_bus_params,
            pred_master_bus_params,
        ) = self.model(tracks_b, ref_mix_a, track_padding_mask=track_padding)

        # ------- generate a mix using the predicted mix console parameters -------
        (
            pred_mixed_tracks_b,
            pred_mix_b,
            pred_track_param_dict,
            pred_fx_bus_param_dict,
            pred_master_bus_param_dict,
        ) = self.mix_console(
            tracks_b,
            pred_track_params,
            pred_fx_bus_params,
            pred_master_bus_params,
            use_track_input_fader=self.use_track_input_fader,
            use_track_panner=self.use_track_panner,
            use_track_eq=self.use_track_eq,
            use_track_compressor=self.use_track_compressor,
            use_fx_bus=self.use_fx_bus,
            use_master_bus=self.use_master_bus,
            use_output_fader=self.use_output_fader,
        )

        # normalize the predicted mix before computing the loss
        # pred_mix_b = batch_stereo_peak_normalize(pred_mix_b)


        if ref_track_param_dict is None:
            ref_track_param_dict = pred_track_param_dict
            ref_fx_bus_param_dict = pred_fx_bus_param_dict
            ref_master_bus_param_dict = pred_master_bus_param_dict

        # ---------------------------- compute and log loss ------------------------------

        
        #print("pred_mix: ", pred_mix_b)
        # if pred_mix_b.sum() == 0:

            #print("pred_track_params: ", pred_track_params)
            #print("pred_fx_bus_params: ", pred_fx_bus_params)
            #print("pred_master_bus_params: ", pred_master_bus_params)
        #print("ref_mix: ",ref_mix_b) 

        loss = 0

        #if parameter_loss is being used to train model, no need to generate mix
        if self.use_param_loss:
            track_param_loss = self.loss(pred_track_params, ref_mix_params)
            loss += track_param_loss
            if self.use_fx_bus:
                fx_bus_param_loss = self.loss(pred_fx_bus_params, ref_fx_bus_params)
                loss += fx_bus_param_loss
            if self.use_master_bus:
                master_bus_param_loss = self.loss(pred_master_bus_params, ref_master_bus_params)
                loss += master_bus_param_loss


        # ---------------------------- compute and log loss ------------------------------

        loss = 0
        if self.use_mix_loss:
            mix_loss = self.loss(pred_mix_b, ref_mix_b)

            if type(mix_loss) == dict:
                for key, val in mix_loss.items():
                    loss += val.mean()
            else:
                loss += mix_loss


            if type(mix_loss) == dict:
                for key, value in mix_loss.items():
                    self.log(
                        ("train" if train else "val") + "/" + key,
                        value,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        sync_dist=True,
                    )
            #print(loss)


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


        # sisdr_error = -self.sisdr(pred_mix_b, ref_mix_b)
        # log the SI-SDR error
        # self.log(
        #    ("train" if train else "val") + "/si-sdr",
        #    sisdr_error,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=False,
        #    logger=True,
        #    sync_dist=True,
        # )

        # mrstft_error = self.mrstft(pred_mix_b, ref_mix_b)
        ## log the MR-STFT error
        # self.log(
        #    ("train" if train else "val") + "/mrstft",
        #    mrstft_error,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=False,
        #    logger=True,
        #    sync_dist=True,
        # )

        # for plotting down the line
        sum_mix_b = tracks_b.sum(dim=1, keepdim=True).detach().float().cpu()
        sum_mix_b = batch_stereo_peak_normalize(sum_mix_b)
        data_dict = {
            "ref_mix_a": ref_mix_a.detach().float().cpu(),
            "ref_mix_b_norm": ref_mix_b.detach().float().cpu(),
            "pred_mix_b_norm": pred_mix_b.detach().float().cpu(),
            "sum_mix_b": sum_mix_b,
            "ref_track_param_dict": ref_track_param_dict,
            "pred_track_param_dict": pred_track_param_dict,
            "ref_fx_bus_param_dict": ref_fx_bus_param_dict,
            "pred_fx_bus_param_dict": pred_fx_bus_param_dict,
            "ref_master_bus_param_dict": ref_master_bus_param_dict,
            "pred_master_bus_param_dict": pred_master_bus_param_dict,
        }
        
        return loss, data_dict

    def training_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=True)

        #print(loss)
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
            #print(optimizer)
            return optimizer
        lr_schedulers = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

        return [optimizer], lr_schedulers
