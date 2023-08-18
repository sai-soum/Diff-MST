import os
import glob
import json
import torch
import yaml
import random
import itertools
import torchaudio
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
from tqdm import tqdm
from typing import List


class MultitrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        metadata_files: List[str],
        instrument_id_json: str = "./data/instrument_name2id.json",
        sample_rate: int = 44100,
        length: int = 524288,
        min_tracks: int = 4,
        max_tracks: int = 20,
        subset: str = "train",
        buffer_reload_rate: int = 4000,
        num_examples_per_epoch: int = 10000,
        buffer_size_gb: float = 0.01,
        target_track_lufs_db: float = -32.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.length = length
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.subset = subset
        self.buffer_reload_rate = buffer_reload_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.target_track_lufs_db = target_track_lufs_db
        self.meter = pyln.Meter(sample_rate)
        self.buffer_frames = self.length

        with open(instrument_id_json, "r") as f:
            self.instrument_ids = json.load(f)

        self.song_dirs = {}
        self.dirs = []

        for directory, split in zip(root_dirs, metadata_files):
            with open(split, "r") as f:
                data = yaml.safe_load(f)
                for songs, track_info in data[self.subset].items():
                    full_song_dir = directory + songs
                    self.song_dirs[full_song_dir] = track_info
                    self.dirs.append(full_song_dir)

        print(len(self.dirs))

        self.items_since_load = self.buffer_reload_rate

    def __len__(self):
        return self.num_examples_per_epoch

    def reload_buffer(self):
        self.examples = []  # clear buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        random.shuffle(self.dirs)  # shuffle dataset
        # load files into RAM
        pbar = tqdm(itertools.cycle(self.dirs))

        for dirname in pbar:
            track_filepaths = glob.glob(os.path.join(dirname, "*.wav"))
            if len(track_filepaths) < self.min_tracks:
                continue
            random.shuffle(track_filepaths)

            num_frames = torchaudio.info(track_filepaths[0]).num_frames

            if num_frames < self.length:
                continue

            # load tracks
            tracks = []
            track_idx = 0
            track_metadata = []
            stereo_info = []
            offset = np.random.randint(
                0.25 * num_frames, num_frames - self.buffer_frames - 1
            )

            for track_filepath in track_filepaths:
                stereo = False
                track, _ = torchaudio.load(
                    track_filepath, frame_offset=offset, num_frames=self.buffer_frames
                )

                if track.shape[-1] != self.buffer_frames:
                    continue
                if track.size()[0] > 2:
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

                instrument = self.song_dirs[dirname][os.path.basename(track_filepath)]
                instrument = self.instrument_ids[instrument]

                if track.size()[0] == 2:
                    stereo = True

                for ch_idx in range(track.shape[0]):
                    if track_idx == self.max_tracks:
                        break
                    else:
                        tracks.append(track[ch_idx : ch_idx + 1, :])
                        track_metadata.append(instrument)
                        if stereo:
                            stereo_info.append(1)
                            stereo = False
                        else:
                            stereo_info.append(0)
                        track_idx += 1

                if track_idx >= self.max_tracks:
                    break

            if track_idx < self.min_tracks:
                continue

            # pad tracks to max_tracks
            while track_idx < self.max_tracks:
                tracks.append(torch.zeros_like(tracks[0]))
                track_metadata.append(0)
                stereo_info.append(0)
                track_idx += 1

            # convert to tensor
            tracks = torch.cat(tracks)
            # tracks = tracks.reshape(self.max_tracks, self.buffer_frames)
            track_metadata = torch.tensor(track_metadata)
            stereo_info = torch.tensor(stereo_info).reshape(track_metadata.shape)

            # add to buffer
            self.examples.append((tracks, stereo_info, track_metadata))

            nbytes_loaded += tracks.element_size() * tracks.nelement()
            pbar.set_description(
                f"Loaded {nbytes_loaded/1e9:0.3f}/{self.buffer_size_gb} gb ({(nbytes_loaded/1e9/self.buffer_size_gb)*100:0.3f}%)"
            )

            # check if buffer is full
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __getitem__(self, _):
        self.items_since_load += 1

        if self.items_since_load >= self.buffer_reload_rate:
            self.reload_buffer()

        example_idx = np.random.randint(0, len(self.examples))
        example = self.examples[example_idx]

        tracks = example[0]
        stereo_info = example[1]
        track_metadata = example[2]

        return tracks, stereo_info, track_metadata


class MultitrackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dirs: List[str],
        metadata_files: List[str],
        length: int,
        min_tracks: int = 4,
        max_tracks: int = 20,
        num_workers: int = 4,
        batch_size: int = 16,
        train_buffer_size_gb: float = 0.01,
        val_buffer_size_gb: float = 0.1,
        num_examples_per_epoch: int = 10000,
        target_track_lufs_db: float = -32.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.current_epoch = -1
        self.max_tracks = self.hparams.min_tracks

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        # count number of times setup is called
        self.current_epoch += 1

        # set the max number of tracks (increase every 10 epochs)
        self.max_tracks = (self.current_epoch // 10) + self.max_tracks

        # cap max tracks at max_tracks
        if self.max_tracks > self.hparams.max_tracks:
            self.max_tracks = self.hparams.max_tracks

        # if self.global_step == self.hparams.active_eq_step:
        #    print("EQ is now active")
        #    self.max_tracks = self.hparams.min_tracks
        # if self.global_step == self.hparams.active_compressor_step:
        #    print("Compressor is now active")
        #    self.max_tracks = self.hparams.min_tracks
        # if self.global_step == self.hparams.active_fx_bus_step:
        #    print("FX bus is now active")
        #    self.max_tracks = self.hparams.min_tracks
        # if self.global_step == self.hparams.active_master_bus_step:
        #    print("Master bus is now active")
        #    self.max_tracks = self.hparams.min_tracks

        print(f"Current epoch: {self.current_epoch} with max_tracks: {self.max_tracks}")

        self.train_dataset = MultitrackDataset(
            root_dirs=self.hparams.root_dirs,
            metadata_files=self.hparams.metadata_files,
            subset="train",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.max_tracks,
            length=self.hparams.length,
            buffer_size_gb=self.hparams.train_buffer_size_gb,
            num_examples_per_epoch=self.hparams.num_examples_per_epoch,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
        )

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_dataset = MultitrackDataset(
            root_dirs=self.hparams.root_dirs,
            metadata_files=self.hparams.metadata_files,
            subset="val",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.max_tracks,
            length=self.hparams.length,
            buffer_size_gb=self.hparams.val_buffer_size_gb,
            num_examples_per_epoch=int(self.hparams.num_examples_per_epoch / 10),
            target_track_lufs_db=self.hparams.target_track_lufs_db,
        )

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        self.test_dataset = MultitrackDataset(
            root_dirs=self.hparams.root_dirs,
            subset="test",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.max_tracks,
            length=self.hparams.length,
            buffer_size_gb=self.hparams.test_buffer_size_gb,
            num_examples_per_epoch=int(self.hparams.num_examples_per_epoch / 10),
            target_track_lufs_db=self.hparams.target_track_lufs_db,
        )

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
        )


# if __name__ == "__main__":

#     dataset = multitrack_dataset()
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

#     for i, (tracks, stereo_info, track_metadata) in enumerate(dataloader):
#         print(tracks.shape)
#         print(stereo_info.shape)
#         print(track_metadata.shape)
#         break
