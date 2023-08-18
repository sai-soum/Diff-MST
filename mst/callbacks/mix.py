import os
import glob
import torch
import wandb
import torchaudio
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
from typing import List


class LogReferenceMix(pl.callbacks.Callback):
    def __init__(
        self,
        root_dirs: List[str],
        ref_mixes: List[str],
        peak_normalize: bool = True,
        sample_rate: int = 44100,
        length: int = 524288,
        target_track_lufs_db: float = -32.0,
    ):
        super().__init__()
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate
        self.length = length
        self.target_track_lufs_db = target_track_lufs_db
        self.meter = pyln.Meter(self.sample_rate)

        self.songs = []
        for root_dir, ref_mix in zip(root_dirs, ref_mixes):
            song = {}
            song["name"] = os.path.basename(root_dir)

            # load reference mix
            x, sr = torchaudio.load(ref_mix)

            # convert sample rate if needed
            if sr != sample_rate:
                x = torchaudio.functional.resample(x, sr, sample_rate)

            song["ref_mix"] = x

            # load tracks
            track_filepaths = glob.glob(os.path.join(root_dir, "*.wav"))
            tracks = []
            for track_filepath in track_filepaths:
                x, sr = torchaudio.load(track_filepath)

                # convert sample rate if needed
                if sr != sample_rate:
                    x = torchaudio.functional.resample(x, sr, sample_rate)

                # separate channels
                for ch_idx in range(x.shape[0]):
                    x_ch = x[ch_idx : ch_idx + 1, : self.length]

                    # loudness normalize
                    track_lufs_db = self.meter.integrated_loudness(
                        x_ch.permute(1, 0).numpy()
                    )

                    if track_lufs_db < -48.0 or track_lufs_db == float("-inf"):
                        continue

                    delta_lufs_db = torch.tensor(
                        [self.target_track_lufs_db - track_lufs_db]
                    ).float()

                    gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                    x_ch = gain_lin * x_ch

                    # save
                    tracks.append(x_ch)

            song["tracks"] = torch.cat(tracks)
            self.songs.append(song)

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
    ):
        """Called when the validation batch ends."""
        for idx, song in enumerate(self.songs):
            ref_mix = song["ref_mix"]
            tracks = song["tracks"]
            name = song["name"]

            # take a chunk from the centre of the tracks and reference mix
            start_idx = (tracks.shape[-1] // 2) - (262144 // 2)
            stop_idx = start_idx + 262144
            tracks_chunk = tracks[..., start_idx:stop_idx]
            ref_mix_chunk = ref_mix[..., start_idx:stop_idx]

            # move to gpu
            tracks_chunk = tracks_chunk.cuda()
            ref_mix_chunk = ref_mix_chunk.cuda()

            with torch.no_grad():
                # predict parameters using the chunks
                (
                    pred_track_params,
                    pred_fx_bus_params,
                    pred_master_bus_params,
                ) = pl_module.model(
                    tracks_chunk.unsqueeze(0), ref_mix_chunk.unsqueeze(0)
                )

                # generate a mix with full tracks using the predicted mix console parameters
                (
                    pred_mixed_tracks,
                    pred_mix_chunk,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = pl_module.mix_console(
                    tracks_chunk.unsqueeze(0),
                    pred_track_params,
                    pred_fx_bus_params,
                    pred_master_bus_params,
                    use_track_gain=pl_module.use_track_gain,
                    use_track_panner=pl_module.use_track_panner,
                    use_track_eq=pl_module.use_track_eq,
                    use_track_compressor=pl_module.use_track_compressor,
                    use_fx_bus=pl_module.use_fx_bus,
                    use_master_bus=pl_module.use_master_bus,
                )

            pred_mix_chunk = pred_mix_chunk.squeeze(0).cpu()
            ref_mix_chunk = ref_mix_chunk.squeeze(0).cpu()

            total_samples = int(
                pred_mix_chunk.shape[-1]
                + ref_mix_chunk.shape[-1]
                + pl_module.mix_console.sample_rate
            )
            y = torch.zeros(total_samples, 2)
            name = f"{idx}_{name}"
            start = 0
            for x, key in zip([ref_mix_chunk, pred_mix_chunk], ["ref_mix", "pred_mix"]):
                end = start + x.shape[-1]
                y[start:end, :] = x.T
                start = end + int(pl_module.mix_console.sample_rate)
                name += key + "-"

            trainer.logger.experiment.log(
                {
                    f"{name}": wandb.Audio(
                        y.numpy(),
                        sample_rate=int(pl_module.mix_console.sample_rate),
                    )
                }
            )
