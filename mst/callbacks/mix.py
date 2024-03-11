import os
import glob
import torch
import wandb
import torchaudio
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
from tqdm import tqdm
from typing import List

from mst.utils import batch_stereo_peak_normalize
from mst.mixing import naive_random_mix


class LogReferenceMix(pl.callbacks.Callback):
    def __init__(
        self,
        root_dirs: List[str],
        ref_mixes: List[str],
        peak_normalize: bool = True,
        sample_rate: int = 44100,
        length: int = 524288,
        target_track_lufs_db: float = -48.0,
        target_mix_lufs_db: float = -16.0,
    ):
        super().__init__()
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate
        self.length = length
        self.target_track_lufs_db = target_track_lufs_db
        self.target_mix_lufs_db = target_mix_lufs_db
        self.meter = pyln.Meter(self.sample_rate)

        print(f"Initalizing reference mix logger with {len(root_dirs)} mixes.")

        self.songs = []
        for root_dir, ref_mix in zip(root_dirs, ref_mixes):

            print(f"Loading {root_dir}...")
            song = {}
            song["name"] = os.path.basename(root_dir)
            

            # load reference mix
            x, sr = torchaudio.load(ref_mix)
            print(f"Reference mix sample rate: {sr}")


            # convert sample rate if needed
            if sr != sample_rate:
                x = torchaudio.functional.resample(x, sr, sample_rate)

            print(f"Reference mix sample rate after resampling: {sample_rate}")


            song["ref_mix"] = x

            # load tracks
            track_filepaths = glob.glob(os.path.join(root_dir, "*.wav"))
            tracks = []
            print("Loading tracks...")
            for track_idx, track_filepath in enumerate(tqdm(track_filepaths)):
                x, sr = torchaudio.load(track_filepath)

                # convert sample rate if needed
                if sr != sample_rate:
                    x = torchaudio.functional.resample(x, sr, sample_rate)

                # separate channels
                for ch_idx in range(x.shape[0]):
                    x_ch = x[ch_idx : ch_idx + 1, :]

                    # save
                    tracks.append(x_ch)

            song["tracks"] = tracks
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

            # take a chunk from the middle of the mix
            start_idx = (ref_mix.shape[-1] // 2) - (131072 // 2)
            stop_idx = start_idx + 131072
            ref_mix_chunk = ref_mix[..., start_idx:stop_idx]

            # loudness normalize the mix
            mix_lufs_db = self.meter.integrated_loudness(
                ref_mix_chunk.permute(1, 0).numpy()
            )
            delta_lufs_db = torch.tensor(
                [self.target_mix_lufs_db - mix_lufs_db]
            ).float()
            gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
            ref_mix_chunk = gain_lin * ref_mix_chunk

            # move to gpu
            ref_mix_chunk = ref_mix_chunk.cuda()

            # make a mix of multiple sections of the tracks
            for n, start_idx in enumerate([0, 524288, 2 * 524288, 3 * 524288]):
                stop_idx = start_idx + 131072

                # loudness normalize tracks
                normalized_tracks = []
                for track in tracks:
                    track = track[..., start_idx:stop_idx]

                    if len(normalized_tracks) > 16:
                        break

                    if track.shape[-1] < 131072:
                        continue

                    track_lufs_db = self.meter.integrated_loudness(
                        track.permute(1, 0).numpy()
                    )

                    if track_lufs_db < -48.0 or track_lufs_db == float("-inf"):
                        continue

                    delta_lufs_db = torch.tensor(
                        [self.target_track_lufs_db - track_lufs_db]
                    ).float()

                    gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                    track = gain_lin * track
                    normalized_tracks.append(track)

                if len(normalized_tracks) == 0:
                    continue

                # cat tracks
                tracks_chunk = torch.cat(normalized_tracks, dim=0)
                tracks_chunk = tracks_chunk.cuda()

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
                        use_track_input_fader=pl_module.use_track_input_fader,
                        use_track_panner=pl_module.use_track_panner,
                        use_track_eq=pl_module.use_track_eq,
                        use_track_compressor=pl_module.use_track_compressor,
                        use_fx_bus=pl_module.use_fx_bus,
                        use_master_bus=pl_module.use_master_bus,
                        use_output_fader=pl_module.use_output_fader,
                    )

                # normalize predicted mix
                pred_mix_chunk = batch_stereo_peak_normalize(pred_mix_chunk)

                # move back to cpu
                pred_mix_chunk = pred_mix_chunk.squeeze(0).cpu()
                ref_mix_chunk_out = ref_mix_chunk.squeeze(0).cpu()

                # generate sum mix
                sum_mix = tracks_chunk.unsqueeze(0).sum(dim=1, keepdim=True).cpu()
                sum_mix = batch_stereo_peak_normalize(sum_mix)
                sum_mix = sum_mix.squeeze(0)

                # generate random mix
                results = naive_random_mix(
                    tracks_chunk.unsqueeze(0),
                    pl_module.mix_console,
                    use_track_input_fader=pl_module.use_track_input_fader,
                    use_track_panner=pl_module.use_track_panner,
                    use_track_eq=pl_module.use_track_eq,
                    use_track_compressor=pl_module.use_track_compressor,
                    use_fx_bus=pl_module.use_fx_bus,
                    use_master_bus=pl_module.use_master_bus,
                    use_output_fader=pl_module.use_output_fader,
                )
                rand_mix = results[1]
                rand_mix = batch_stereo_peak_normalize(rand_mix).cpu()
                rand_mix = rand_mix.squeeze(0)

                audios = {
                    "ref_mix": ref_mix_chunk_out,
                    "pred_mix": pred_mix_chunk,
                    "sum_mix": sum_mix,
                    "rand_mix": rand_mix,
                }

                total_samples = 0
                for x in audios.values():
                    total_samples += x.shape[-1] + int(
                        pl_module.mix_console.sample_rate
                    )

                y = torch.zeros(total_samples, 2)
                log_name = f"{idx}_{n}{name}"
                start = 0
                for key, x in audios.items():
                    end = start + x.shape[-1]
                    y[start:end, :] = x.T
                    start = end + int(pl_module.mix_console.sample_rate)
                    log_name += key + "-"

                trainer.logger.experiment.log(
                    {
                        f"{log_name}": wandb.Audio(
                            y.numpy(),
                            sample_rate=int(pl_module.mix_console.sample_rate),
                        )
                    }
                )
