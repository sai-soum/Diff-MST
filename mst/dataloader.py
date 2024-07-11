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

from torch.utils.data import random_split


class MixDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, length: int = 524288):
        super().__init__()
        self.root_dir = root_dir
        self.length = length

         
        self.mix_filepaths = glob.glob(
            os.path.join(root_dir, "**", "*.wav"), recursive=True)

        #self.mix_filepaths = glob.glob(
            #os.path.join(root_dir, "**", "*.mp3"), recursive=True)
        print(f"Located {len(self.mix_filepaths)} mixes.")

        self.meter = pyln.Meter(44100)

    def __len__(self):
        return len(self.mix_filepaths)

    def __getitem__(self, _):
        valid = False
        while not valid:
            # get random file
            idx = np.random.randint(0, len(self.mix_filepaths))
            # idx = 42  # always use the same mix for debug
            mix_filepath = self.mix_filepaths[idx]
            num_frames = torchaudio.info(mix_filepath).num_frames

            # find random non-silent region of the mix
            offset = np.random.randint(0, num_frames - self.length - 1)

            offset = 0  # always use the same offset


            mix, _ = torchaudio.load(
                mix_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )

            if mix.shape[0] == 1:
                mix = mix.repeat(2, 1)
            elif mix.shape[0] > 2:
                mix = mix[:2, :]

            if mix.shape[-1] != self.length:
                continue

            mix_lufs_db = self.meter.integrated_loudness(mix.permute(1, 0).numpy())

            if mix_lufs_db > -48.0:
                valid = True

            # random gain of the target mixes
            target_lufs_db = np.random.randint(-48, 0)
            target_lufs_db = -14.0  # always use same target
            delta_lufs_db = torch.tensor([target_lufs_db - mix_lufs_db]).float()
            mix = 10.0 ** (delta_lufs_db / 20.0) * mix

        return mix


class MixDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        length: int = 524288,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()
        torchaudio.set_audio_backend("soundfile")

    def setup(self, stage=None):
        # create dataset
        dataset = MixDataset(self.hparams.root_dir, self.hparams.length)
        # create random splits
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [0.8, 0.1, 0.1]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=1,
        )


class MultitrackDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        track_root_dirs: List[str],
        metadata_files: List[str],
        mix_root_dirs: List[str] = [],
        instrument_id_json: str = "./data/instrument_name2id.json",
        sample_rate: int = 44100,
        length: int = 524288,
        min_tracks: int = 4,
        max_tracks: int = 20,
        subset: str = "train",
        buffer_size_gb: float = 0.01,
        target_track_lufs_db: float = -32.0,
        target_mix_lufs_db: float = -16.0,
        randomize_ref_mix_gain: bool = False,
        num_examples_per_epoch: int = 20000,
        num_passes: int = 1,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.length = length
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.subset = subset
        self.buffer_size_gb = buffer_size_gb
        self.target_track_lufs_db = target_track_lufs_db
        self.target_mix_lufs_db = target_mix_lufs_db
        self.randomize_ref_mix_gain = randomize_ref_mix_gain
        self.meter = pyln.Meter(sample_rate)
        self.length = self.length
        self.num_passes = num_passes
        self.num_examples_per_epoch = num_examples_per_epoch

        with open(instrument_id_json, "r") as f:
            self.instrument_ids = json.load(f)

        self.song_dirs = {}
        self.dirs = []

        # load metadata for tracks
        for directory, split in zip(track_root_dirs, metadata_files):
            with open(split, "r") as f:
                data = yaml.safe_load(f)
                for songs, track_info in data[self.subset].items():
                    full_song_dir = directory + songs
                    self.song_dirs[full_song_dir] = track_info
                    self.dirs.append(full_song_dir)

        print(f"Located {len(self.dirs)} track directories.")

        # load metadata for mixes
        self.mix_dirs = {}
        self.mixes = []

        for mix_dir in mix_root_dirs:
            # find all mixes in directory recursively

            mix_files = glob.glob(os.path.join(mix_dir, "**", "*.wav"), recursive=True)


            self.mixes.extend(mix_files)

        print(f"Located {len(self.mixes)} mixes.")

        self.num_examples = (
            self.num_examples_per_epoch + 1
        )  # this will trigger a reload of the buffer

    def __len__(self):
        return self.num_examples_per_epoch

    def reload_mix_buffer(self):
        self.mix_examples = []  # clear buffer
        nbytes_loaded = 0  # counter for data in RAM

        random.shuffle(self.mix_dirs)  # shuffle dataset

        pbar = tqdm(itertools.cycle(self.mixes))

        for filepath in pbar:
            num_frames = torchaudio.info(filepath, backend="soundfile").num_frames
            offset = np.random.randint(0.25 * num_frames, num_frames - self.length - 1)

            # ensure the song is long enough if we start from 25% in
            if (0.75 * num_frames) < self.length:
                continue

            mix, _ = torchaudio.load(
                filepath,
                frame_offset=offset,
                num_frames=self.length,
                backend="soundfile",
            )

            if mix.shape[0] == 1:
                continue
            if mix.shape[-1] != self.length:
                continue
            if mix.size()[0] > 2:
                continue

            mix_lufs_db = self.meter.integrated_loudness(mix.permute(1, 0).numpy())



            if mix_lufs_db < -48.0 or mix_lufs_db == float("-inf"):
                continue

            delta_lufs_db = torch.tensor(
                [self.target_mix_lufs_db - mix_lufs_db]
            ).float()

            gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
            mix = gain_lin * mix

            self.mix_examples.append(mix)

            nbytes_loaded += mix.element_size() * mix.nelement()
            pbar.set_description(
                f"Loaded {nbytes_loaded/1e9:0.3f}/{self.buffer_size_gb} gb ({(nbytes_loaded/1e9/self.buffer_size_gb)*100:0.3f}%)"
            )

            # check if buffer is full
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def reload_track_buffer(self):
        self.track_examples = []  # clear buffer
        nbytes_loaded = 0  # counter for data in RAM

        random.shuffle(self.dirs)  # shuffle dataset

        # load files into RAM
        pbar = tqdm(itertools.cycle(self.dirs))

        for dirname in pbar:
            track_filepaths = glob.glob(os.path.join(dirname, "*.wav"))

            song_name = os.path.basename(dirname)

            if len(track_filepaths) < self.min_tracks:
                continue
            random.shuffle(track_filepaths)

            num_frames = torchaudio.info(
                track_filepaths[0], backend="soundfile"
            ).num_frames

            middle_idx = int(num_frames / 2)

            # ensure the song is long enough if we start from 25% in
            if (0.75 * num_frames) < self.length:
                continue

            # load tracks
            tracks = []
            track_idx = 0
            track_metadata = []
            stereo_info = []
            track_padding = []
            # find a starting offset 25% into the song or more
            offset = np.random.randint(0.25 * num_frames, num_frames - self.length - 1)

            for track_filepath in track_filepaths:
                stereo = False
                track, _ = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.length,
                    backend="soundfile",
                )

                if track.shape[-1] != self.length:
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
                        track_padding.append(False)
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
                track_padding.append(True)
                stereo_info.append(0)
                track_idx += 1

            # convert to tensor
            tracks = torch.cat(tracks)

            
            # if tracks[...,0:middle_idx].sum() == 0 or tracks[...,middle_idx:].sum() == 0:
            #     continue
            tracks = tracks.reshape(self.max_tracks, self.length)
            #create a sum mix of the tracks
            mix_check = tracks.sum(0)
            if torch.any(mix_check[...,0:middle_idx] == False) or torch.any(mix_check[...,middle_idx:] == False):
                continue

            track_metadata = torch.tensor(track_metadata)
            stereo_info = torch.tensor(stereo_info).reshape(track_metadata.shape)
            track_padding = torch.tensor(track_padding)

            # add to buffer
            self.track_examples.append(

                (tracks, stereo_info, track_metadata, track_padding, song_name)

            )

            nbytes_loaded += tracks.element_size() * tracks.nelement()
            pbar.set_description(
                f"Loaded {nbytes_loaded/1e9:0.3f}/{self.buffer_size_gb} gb ({(nbytes_loaded/1e9/self.buffer_size_gb)*100:0.3f}%)"
            )

            # check if buffer is full
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __getitem__(self, idx):

        # ----------- reload buffers if needed ------------
        if self.num_examples > self.num_examples_per_epoch:
            self.reload_track_buffer()
            self.reload_mix_buffer()
            self.num_examples = 0  # reset counter

        # ------------ get example from track buffer ------------
        track_example_idx = np.random.randint(0, len(self.track_examples))
        track_example = self.track_examples[track_example_idx]

        tracks = track_example[0]

        
        stereo_info = track_example[1]
        track_metadata = track_example[2]
        track_padding = track_example[3]
        song_name = track_example[4]


        # ------------ get example from mix buffer ------------
        # optional
        if len(self.mix_examples) > 0:
            mix_example_idx = np.random.randint(0, len(self.mix_examples))
            mix = self.mix_examples[mix_example_idx]

            if self.randomize_ref_mix_gain:
                gain_db = np.random.uniform(-16.0, 12.0)
                gain_lin = 10.0 ** (gain_db / 20.0)
                mix = gain_lin * mix
        else:
            mix = torch.empty(1)


        return tracks, stereo_info, track_metadata, track_padding, mix, song_name



class MultitrackDataModule(pl.LightningDataModule):
    def __init__(
        self,
        track_root_dirs: List[str],
        metadata_files: List[str],
        length: int,
        mix_root_dirs: List[str] = [],
        min_tracks: int = 4,
        max_tracks: int = 20,
        num_workers: int = 4,
        batch_size: int = 16,
        num_train_examples: int = 20000,
        num_val_examples: int = 1000,
        num_train_passes: int = 1,
        num_val_passes: int = 1,
        train_buffer_size_gb: float = 0.01,
        val_buffer_size_gb: float = 0.1,
        target_track_lufs_db: float = -48.0,
        target_mix_lufs_db: float = -16.0,
        randomize_ref_mix_gain: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        self.train_dataset = MultitrackDataset(
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.metadata_files,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="train",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.hparams.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_train_passes,
            buffer_size_gb=self.hparams.train_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
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
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.metadata_files,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="val",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.hparams.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_val_passes,
            buffer_size_gb=self.hparams.val_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
        )

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=1,
        )

    def test_dataloader(self):
        self.test_dataset = MultitrackDataset(
            track_root_dirs=self.hparams.track_root_dirs,
            metadata_files=self.hparams.split,
            mix_root_dirs=self.hparams.mix_root_dirs,
            subset="test",
            min_tracks=self.hparams.min_tracks,
            max_tracks=self.max_tracks,
            length=self.hparams.length,
            num_passes=self.hparams.num_val_passes,
            buffer_size_gb=self.hparams.test_buffer_size_gb,
            target_track_lufs_db=self.hparams.target_track_lufs_db,
            target_mix_lufs_db=self.hparams.target_mix_lufs_db,
            randomize_ref_mix_gain=self.hparams.randomize_ref_mix_gain,
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
