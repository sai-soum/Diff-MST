import os
import json
import glob
import yaml
import torch
import random
import itertools
import torchaudio
import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
from yaml import load, dump
from yaml import Loader, Dumper
import json
from tqdm import tqdm
from typing import List

# TODO: use consistent data format (either ymal or json)


class MedleyDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        metadata_dir: str = "./data/medley_metadata",
        instrument_id_json: str = "./data/instrument_name2id.json",
        dataset_split_yaml: str = "./data/medley_split.yaml",
        sample_rate: float = 44100,
        min_tracks: int = 4,
        max_tracks: int = 8,
        length: float = 524288,
        subset: str = "test",
        buffer_reload_rate: int = 4000,
        num_examples_per_epoch: int = 10000,
        buffer_size_gb: float = 0.2,
        target_track_lufs_db: float = -32.0,
    ) -> None:
        super().__init__()
        torchaudio.set_audio_backend("sox_io")
        self.sample_rate = sample_rate
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.length = length
        self.metadata_dir = metadata_dir
        self.subset = subset
        self.buffer_reload_rate = buffer_reload_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.buffer_frames = (
            self.length
        )  # load examples with same size as the train length
        self.target_track_lufs_db = target_track_lufs_db

        self.meter = pyln.Meter(sample_rate)  # create BS.1770 meter

        self.mix_dirs = []
        # We have a dictionary with different numbers assigned to each of the instruments.
        # We will assign these numbers to each of the tracks based on the instrument name in the metadata file.
        # this will let us return this data from the dataloader
        with open(instrument_id_json, "r") as f:
            self.instrument_ids = json.load(f)

        with open(dataset_split_yaml, "r") as f:
            self.medley_split = yaml.safe_load(f)

        for root_dir in root_dirs:
            # find all mix directories
            mix_dirs = glob.glob(os.path.join(root_dir, "*"))
            # remove items that are not directories
            mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
            mix_dirs = sorted(mix_dirs)  # sort
            self.mix_dirs += mix_dirs
        print(f"Found {len(self.mix_dirs)} mix directories in {root_dirs}.")

        subset_dirs = []
        for mix_dir in self.mix_dirs:
            mix_id = os.path.basename(mix_dir)
            if mix_id in self.medley_split[self.subset]:
                subset_dirs.append(mix_dir)
        self.mix_dirs = subset_dirs
        print(f"Found {len(self.mix_dirs)} mix directories in {self.subset} set.")

        filtered_mix_dirs = []

        for mix_dir in tqdm(self.mix_dirs):
            mix_id = os.path.basename(mix_dir)
            track_filepaths = glob.glob(os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav"))
            # remove all mixes that have more tracks than 16 and less than 4 requested
            if (
                len(track_filepaths) <= self.max_tracks
                and len(track_filepaths) >= self.min_tracks
            ):
                filtered_mix_dirs.append(mix_dir)

        self.mix_dirs = filtered_mix_dirs
        print(
            f"Found {len(self.mix_dirs)} mix directories with tracks less than {self.max_tracks} and more than {self.min_tracks}."
        )
        self.items_since_load = self.buffer_reload_rate

    def reload_buffer(self):
        self.examples = []  # clear buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        # different subset in each
        random.shuffle(self.mix_dirs)

        # load files into RAM
        pbar = tqdm(itertools.cycle(self.mix_dirs))
        for mix_dir in pbar:
            mix_id = os.path.basename(mix_dir)
            mix_filepath = glob.glob(os.path.join(mix_dir, "*.wav"))[0]
            metadata_filepath = os.path.join(
                self.metadata_dir, f"{mix_id}_METADATA.yaml"
            )
            with open(metadata_filepath, "r") as f:
                mdata = yaml.safe_load(f)

            if "AimeeNorwich_Child" in mix_filepath:
                continue  # this song is corrupted

            # save only a random subset of this song so we can load more songs
            silent = True
            counter = 0
            while silent:
                num_frames = torchaudio.info(mix_filepath).num_frames
                offset = np.random.randint(0, num_frames - self.buffer_frames - 1)

                # now check the length of the mix
                y, sr = torchaudio.load(
                    mix_filepath,
                    frame_offset=offset,
                    num_frames=self.buffer_frames,
                )

                if (y**2).mean() > 1e-3:
                    silent = False

                counter += 1
                if counter > 10:
                    break

            if silent:
                continue  # skip this song after 10 tries

            if y.shape[-1] != self.buffer_frames:
                continue

            mix_num_frames = y.shape[-1]

            # now find all the track filepaths
            track_filepaths = glob.glob(os.path.join(mix_dir, f"{mix_id}_RAW", "*.wav"))

            # check length of each track
            tracks = []
            metadata = []
            genre = mdata["genre"]
            # print(len(track_filepaths))
            for tidx, track_filepath in enumerate(track_filepaths):
                # load the track
                track, sr = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.buffer_frames,
                )

                if track.shape[-1] != self.buffer_frames:
                    continue  # not sure why we need this yet, but it seems to be necessary

                # loudness normalization
                track_lufs_db = self.meter.integrated_loudness(y.permute(1, 0).numpy())

                if track_lufs_db == float("-inf"):
                    continue

                delta_lufs_db = torch.tensor(
                    [self.target_track_lufs_db - track_lufs_db]
                ).float()
                gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                track = gain_lin * track

                # loudness normalization
                track_lufs_db = self.meter.integrated_loudness(y.permute(1, 0).numpy())

                if track_lufs_db == float("-inf"):
                    continue

                delta_lufs_db = torch.tensor(
                    [self.target_track_lufs_db - track_lufs_db]
                ).float()
                gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                track = gain_lin * track

                tracks.append(track)
                # get the instrumnet name from the metadata file
                instru_id = mdata[os.path.basename(track_filepath)]
                # assign the instrument id based on the instrument name

                inst_id = self.instrument_ids[instru_id]
                metadata.append(inst_id)
                if track.shape[-2] == 2:
                    metadata.append(inst_id)

                nbytes = track.element_size() * track.nelement()
                nbytes_loaded += nbytes

            if len(tracks) < self.min_tracks:
                continue

            # store this example
            example = {
                "mix_id": os.path.dirname(mix_filepath).split(os.sep)[-1],
                "mix_filepath": mix_filepath,
                "num_frames": mix_num_frames,
                "track_filepaths": track_filepaths,
                "tracks": tracks,
                "metadata": metadata,
                "genre": genre,
            }
            for track in tracks:
                if torch.isnan(track).any():
                    raise ValueError("Found nan while loading tracks!")

            self.examples.append(example)
            pbar.set_description(f"Loaded {nbytes_loaded/1e9:0.3} gb")

            # check the size of loaded data
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, _):
        # increment counter
        self.items_since_load += 1

        # load next chunk into buffer if needed
        if self.items_since_load > self.buffer_reload_rate:
            self.reload_buffer()

        # select an example at random
        example_idx = np.random.randint(0, len(self.examples))
        example = self.examples[example_idx]

        # read tracks from RAM
        tracks = torch.zeros(self.max_tracks, self.length)
        instrument_id = example["metadata"]

        stereo_info = []
        track_idx = 0

        num_silent_tracks = self.max_tracks - len(instrument_id)
        for i in range(num_silent_tracks):
            instrument_id.append(0)
            i += 1

        stereo = 0
        for track in example["tracks"]:
            if track.shape[0] == 2:
                stereo = 1
            # print(track.shape)
            # print(len(instrument_id))
            for ch_idx in range(track.shape[0]):
                if track_idx == self.max_tracks:
                    break
                else:
                    tracks[track_idx, :] = track[ch_idx, :]
                    track_idx += 1
                    if stereo == 1:
                        stereo_info.append(stereo)
                        stereo = 0
                    else:
                        stereo_info.append(stereo)

        # print(tracks.shape)
        if len(stereo_info) != len(instrument_id):
            for i in range(len(instrument_id) - len(stereo_info)):
                stereo_info.append(0)

        instrument_id = torch.tensor(instrument_id)
        stereo_info = torch.tensor(stereo_info).reshape(instrument_id.shape)

        return tracks, instrument_id, stereo_info


class MedleyDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dirs: List[str],
        length: int,
        min_tracks: int = 4,
        max_tracks: int = 20,
        num_workers: int = 4,
        batch_size: int = 16,
        train_buffer_size_gb: float = 2.0,
        val_buffer_size_gb: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MedleyDBDataset(
                root_dirs=self.hparams.root_dirs,
                subset="train",
                min_tracks=self.hparams.min_tracks,
                max_tracks=self.hparams.max_tracks,
                length=self.hparams.length,
                buffer_size_gb=self.hparams.train_buffer_size_gb,
                num_examples_per_epoch=10000,
            )

        if stage == "validate" or stage == "fit":
            self.val_dataset = MedleyDBDataset(
                root_dirs=self.hparams.root_dirs,
                subset="val",
                min_tracks=self.hparams.min_tracks,
                max_tracks=self.hparams.max_tracks,
                length=self.hparams.length,
                buffer_size_gb=self.hparams.val_buffer_size_gb,
                num_examples_per_epoch=1000,
            )

        if stage == "test":
            self.test_dataset = MedleyDBDataset(
                root_dirs=self.hparams.root_dirs,
                subset="test",
                min_tracks=self.hparams.min_tracks,
                max_tracks=self.hparams.max_tracks,
                length=self.hparams.length,
                buffer_size_gb=self.hparams.test_buffer_size_gb,
                num_examples_per_epoch=1000,
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
