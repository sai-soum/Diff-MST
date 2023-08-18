import os
import glob
import torch
import wandb
import torchaudio
import numpy as np
import pytorch_lightning as pl
from typing import List


class LogReferenceMix(pl.callbacks.Callback):
    def __init__(
        self,
        root_dirs: List[str],
        ref_mixes: List[str],
        peak_normalize: bool = True,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate

        self.songs = []
        for root_dir, ref_mix in zip(root_dirs, ref_mixes):
            song = {}
            song["name"] = os.path.basename(root_dir)

            # load reference mix
            x, sr = torchaudio.load(ref_mix)
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
                    x_ch = x[ch_idx, :]

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
                    track = gain_lin * track

                    # save
                    tracks.append(x_ch)

            song["tracks"] = tracks
            self.songs.append(song)

    def on_validation_epoch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        """Called when the validation batch ends."""
        for idx, song in enumerate(self.songs):
            ref_mix = song["ref_mix"]
            tracks = song["tracks"]
            caption = song["name"]
            with torch.no_grad():
                result = pl_module.model(
                    tracks,
                    ref_mix,
                    use_track_gain=True,
                    use_track_panner=True,
                    use_track_eq=pl_module.use_track_eq,
                    use_track_compressor=pl_module.use_track_compressor,
                    use_fx_bus=pl_module.use_fx_bus,
                    use_master_bus=pl_module.use_master_bus,
                )

                (
                    pred_mix_tracks,
                    pred_mix,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = result

            total_samples = (
                pred_mix.shape[-1] + song["ref_mix"].shape[-1] + pl_module.sample_rate
            )
            y = torch.zeros(total_samples, 2)
            name = f"{idx}_"
            start = 0
            for x, key in zip([ref_mix, pred_mix], ["ref_mix", "pred_mix"]):
                end = start + x.shape[0]
                y[start:end, :] = x
                start = end + int(pl_module.sample_rate)
                name += key + "-"

            trainer.logger.experiment.log(
                {
                    f"{name}": wandb.Audio(
                        y.numpy(),
                        caption=caption,
                        sample_rate=int(pl_module.sample_rate),
                    )
                }
            )
