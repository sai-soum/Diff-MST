import os
import glob
import torch
import random
import itertools
import torchaudio
import numpy as np
import pytorch_lightning as pl
from yaml import load, dump
from yaml import Loader, Dumper
import json
from tqdm import tqdm
from typing import List



class MedleyDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        metadata_dir = "/homes/ssv02/Diff-MST/medley_metadata",
        sample_rate: float = 44100,
        min_tracks: int = 4,
        max_tracks: int = 20,
        length: float = 524288,
        indices: List[int] = [0, 90],
        buffer_reload_rate: int = 4000,
        num_examples_per_epoch: int = 1000,
        buffer_size_gb: float = 0.02,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.length = length
        self.metadata_dir = metadata_dir
        self.indices = indices
        self.buffer_reload_rate = buffer_reload_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.buffer_frames = (
            self.length
        )  # load examples with same size as the train length

        self.mix_dirs = []
        #We have a dictionary with different numbers assigned to each of the instruments. 
        #We will assign these numbers to each of the tracks based on the instrument name in the metadata file.
        #this will let us return this data from the dataloader
        instrument_ids = json.load(open("/homes/ssv02/Diff-MST/inst_id.txt"))
        self.instrument_ids = instrument_ids
        #print(type(self.instrument_ids))
        
       


        for root_dir in root_dirs:
            # find all mix directories
            mix_dirs = glob.glob(os.path.join(root_dir, "*"))
            # remove items that are not directories
            mix_dirs = [mix_dir for mix_dir in mix_dirs if os.path.isdir(mix_dir)]
            mix_dirs = sorted(mix_dirs)  # sort
            self.mix_dirs += mix_dirs
        print(f"Found {len(self.mix_dirs)} mix directories in {root_dirs}.")
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
        self.mix_dirs = self.mix_dirs[indices[0] : indices[1]]  # select subset
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
            print(mix_id)
            mix_filepath = glob.glob(os.path.join(mix_dir, "*.wav"))[0]
            metadata_filepath = os.path.join(self.metadata_dir, f"{mix_id}_METADATA.yaml")
            with open(metadata_filepath, 'r') as f:
                mdata = load(f, Loader=Loader)
            
            if mdata.values() in ['Main system']:
                continue
            
            # print(mix_filepath)
            if "AimeeNorwich_Child" in mix_filepath:
                continue
            #print(mdata)
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
           
            # print("\nMetadata:", mdata)
            # for track_filepath in track_filepaths:
            #     file= os.path.basename(track_filepath)
            #     instru_id = mdata[file]
            #     print(file, instru_id)
            
            # check length of each track
            tracks = []
            metadata = []
            genre = mdata['genre']
            #print(len(track_filepaths))
            for tidx, track_filepath in enumerate(track_filepaths):
                # load the track
                track, sr = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.buffer_frames,
                )
                #-------------------------------------------
                
                #this checks for silence in the track
                # if (track**2).mean() < 1e-3:
                #     print("passed")
                #     continue
                #-------------------------------------------
                #print(track_filepath)
                # if track.shape[-1] != self.buffer_frames:
                #     continue

                tracks.append(track)
                #get the instrumnet name from the metadata file
                instru_id = mdata[os.path.basename(track_filepath)]
                #assign the instrument id based on the instrument name
                inst_id = self.instrument_ids[instru_id]
                
                #print(instru_id)
                metadata.append(inst_id)
                # print("\n")
                # print(os.path.basename(track_filepath), mdata[os.path.basename(track_filepath)])
                nbytes = track.element_size() * track.nelement()
                nbytes_loaded += nbytes
            #print("\n")
            print(metadata)
            if len(metadata) != len(tracks):
                print("Metadata and tracks do not match")
                exit()
            # store this example
            example = {
                "mix_id": os.path.dirname(mix_filepath).split(os.sep)[-1],
                "mix_filepath": mix_filepath,
                "num_frames": mix_num_frames,
                "track_filepaths": track_filepaths,
                "tracks": tracks,
                "metadata": metadata,
                "genre": genre
            }

            self.examples.append(example)
            pbar.set_description(f"Loaded {nbytes_loaded/1e9:0.3} gb")

            # check the size of loaded data
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

        # print("Number of songs loaded into buffer:", len(self.examples))

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

        
        track_idx = 0
        
        num_silent_tracks = self.max_tracks-len(instrument_id)
        for i in range(num_silent_tracks):
            instrument_id.append(0)
            i += 1
        
        
        
        for track in example["tracks"]:
            # print(track.shape)
            # print(len(instrument_id))
            for ch_idx in range(track.shape[0]):
                if track_idx > self.max_tracks:
                    break
                else:
                    tracks[track_idx, :] = track[ch_idx, :]
                    track_idx += 1
        # print(tracks.shape)
        
        instrument_id = torch.tensor(instrument_id)
        
        return tracks, instrument_id, example["genre"]


class MedleyDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dirs: List[str],
        length: int,
        num_workers: int = 4,
        batch_size: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = MedleyDBDataset(
                root_dirs=self.hparams.root_dirs,
                indices=[0, 90],
                length=self.hparams.length,
            )

        if stage == "validate" or stage == "fit":
            self.val_dataset = MedleyDBDataset(
                root_dirs=self.hparams.root_dirs,
                indices=[90, 100],
                length=self.hparams.length,
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
            num_workers=4,
        )

# if __name__ == "__main__":
#     dataset = MedleyDBDataset(root_dirs=["/import/c4dm-datasets/MedleyDB_V1/V1"])
#     print(len(dataset))

#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

#     for i, (track, instrument_id, genre) in enumerate(dataloader):
#         #print(track.shape)
#         #print(track.shape)
#         # print(instrument_id)
#         # print(genre)
#         # print(track.size())
        

#         if i == 0:

#             break