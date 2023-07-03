import os
import glob
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
import string
from os.path import dirname as up
from mst.dataloaders.musdb import ToMono
from string import digits


def extract_metadata(track_names: list):
    """Extract metadata from track names.

    Args:
        track_names (list): List of track names.

    Returns:
        dict: Dictionary of metadata.
    """
    metadata_inst = [ "piano", "guitar", "bass", "vocals",  "kick","snare","vox", "woodwind","cabasa","horn", "percussion","shaker", "quartet","tom", "fiddle","noise","vocoder","soprano", "alto", "tenor","loop", "bell", "hi-hat", "hi hat", "chorus", " gong", "brass", "hit", "rhodes", "overhead","cajon","crash", "stick", "cymbal", "sfx", "hihat", "synth","string","clap","ukelele","bagpipe","whistle", "organ", "flute", "saxophone", "violin", "cello", "trumpet", "trombone", "clarinet", "conga","tuba", "harp", "accordion","keys", "banjo", "mandolin", "harmonica", "xylophone", "glockenspiel", "bassoon", "oboe", "sitar","drum", "ukulele", "viola", "tambourine", "marimba", "triangle", "bagpipes", "bongos", "steelpan", "theremin", "tambourine", "harpsichord", "timpani", "tuba", "viola", "violin","elecgtr", "gtr" ,"cello", "trumpet", "trombone", "clarinet", "tuba", "harp", "accordion", "banjo", "mandolin", "harmonica", "xylophone", "glockenspiel", "bassoon", "oboe", "sitar", "ukulele", "viola", "tambourine", "marimba", "triangle", "bagpipes", "bongos", "steelpan", "theremin", "tambourine", "harpsichord", "timpani", "tuba", "viola", "violin", "cello", "trumpet", "trombone", "clarinet", "tuba", "harp", "accordion", "banjo", "mandolin", "harmonica", "xylophone", "glockenspiel", "bassoon", "oboe", "sitar", "ukulele", "viola", "tambourine", "marimba", "triangle", "bagpipes", "bongos", "steelpan", "theremin", "tambourine", "harpsichord", "timpani", "tuba", "viola", "violin", "cello", "trumpet", "trombone", "clarinet", "tuba", "harp", "accordion", "banjo", "mandolin", "harmonica", "xylophone", "glockenspiel", "bassoon", "oboe", "sitar", "ukulele", "choir" ]

    tracks = [os.path.basename(track) for track in track_names]
    instrument_id = [track.split("/")[-1].split(".")[0] for track in track_names]
    
    remove_digits = str.maketrans('', '', digits)
    
    instrument_id = [id.translate(remove_digits) for id in instrument_id]
    #print(instrument_id)
    instrument_id = [id.replace("_", "") for id in instrument_id]
    instrument_id = [id.replace("DT", "") for id in instrument_id]
    instrument_id = [id.replace("DI", "") for id in instrument_id]
    instrument_id = [id.replace("Drumkit", "") for id in instrument_id]
    #print(instrument_id)
    instrument_id = [id.lower() for id in instrument_id]
    #print(instrument_id)

    
    for i, id in enumerate(instrument_id):
        #print(id)
        if id == "backingvox":
            #print("backingvox entered")
            continue
        elif "gtr" in id:
            #print("gtr")
            instrument_id[i]="guitar"
        elif "hat" in id:
            instrument_id[i]="hi hat"
        elif "room" in id:
            instrument_id[i]="roommic"
        elif "sax" in id:
            #print("sax")
            instrument_id[i]="saxophone"
        elif id == "basssynth":
            instrument_id[i]="bass"
        elif "hammond" in id:
            instrument_id[i]="organ"
        elif id in ["guasa", "paliteo"]:
            instrument_id[i] = "misc"
        elif id in ["bombo", "conga", "tambor"]:
            instrument_id[i] = "percussion"
        else:
            ctr = 0
            for meta in metadata_inst:
                if meta in id:
                    #print("meta", meta, "id", id)
                    #print("\n", id, meta)
                    instrument_id[i] = meta
                    ctr = 1
                    break
                    #print(instrument_id[i])
                
                
            if ctr == 0:
                instrument_id[i] = "misc"
    
    metadata_track = dict(zip(tracks, instrument_id))
    return metadata_track


class CambridgeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dirs: List[str],
        sample_rate: float = 44100,
        min_tracks: int = 4,
        max_tracks: int = 20,
        length: float = 524288,
        indices: List[int] = [0,150],
        buffer_reload_rate: int = 4000,
        num_examples_per_epoch: int = 10000,
        buffer_size_gb: float = 0.2,
        target_track_lufs_db: float = -32.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.min_tracks = min_tracks
        self.max_tracks = max_tracks
        self.length = length
        self.buffer_reload_rate = buffer_reload_rate
        self.num_examples_per_epoch = num_examples_per_epoch
        self.buffer_size_gb = buffer_size_gb
        self.buffer_frames = (
            self.length
        )  # load examples with same size as the train length
        self.target_track_lufs_db = target_track_lufs_db
        self.indices = indices
        self.meter = pyln.Meter(sample_rate)  # create BS.1770 meter

        self.mix_dirs = []
        #We have a dictionary with different numbers assigned to each of the instruments. 
        #We will assign these numbers to each of the tracks based on the instrument name in the metadata file.
        #this will let us return this data from the dataloader
       
        
        self.mix_dirs = []
        paths = glob.glob(os.path.join(root_dirs[0], "*"))
        #print(f"Found {len(paths)} mix directories in {root_dirs}.")
        for path in paths:
            
            if os.path.basename(path) == "00275":
                continue
            #print(os.path.basename(path))
            song_dir = os.path.join(path,"tracks/")
            #print(song_dir)
            raw_tracks_dir = [folder for folder in os.listdir(song_dir) if os.path.isdir(os.path.join(song_dir, folder))]
            #print(raw_tracks_dir)
            raw_tracks_dir = [dir for dir in raw_tracks_dir if "_Full" in dir or len(raw_tracks_dir)==1]
            #print(raw_tracks_dir)
            raw_tracks_dir = os.path.join(song_dir, raw_tracks_dir[0])
            if not os.path.exists(raw_tracks_dir):
                #print("No tracks found in " + raw_tracks_dir)
                continue
            else:
                #print(raw_tracks_dir)
                mix_dirs = raw_tracks_dir
                raw_tracks = glob.glob(os.path.join(raw_tracks_dir, "*.wav"))
                #print(f"Found {len(raw_tracks)} tracks in {raw_tracks_dir}.")
                if len(raw_tracks)>=self.min_tracks and len(raw_tracks)<=self.max_tracks:   
                    self.mix_dirs.append(mix_dirs)
                    
            # print(song_dir)
        self.mix_dirs = self.mix_dirs[self.indices[0]:self.indices[1]]
        print(f"{len(self.mix_dirs)} directories found with at least {self.min_tracks} tracks and at most {self.max_tracks} tracks.")
        self.instrument_ids = json.load(open("/homes/ssv02/Diff-MST/inst_id.txt"))
            # print(raw_tracks_dir)
            

       
        self.items_since_load = self.buffer_reload_rate

    def reload_buffer(self):

        stereo_to_mono = ToMono()
        self.examples = []  # clear buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        # different subset in each
        random.shuffle(self.mix_dirs)

        # load files into RAM
        pbar = tqdm(itertools.cycle(self.mix_dirs))


        for mix_dir in pbar:
            #print(mix_dir)
        
            #mix_id = os.path.basename(mix_dir)
            #print(mix_id)
            track_filepaths = glob.glob(os.path.join(mix_dir, "*.wav"))
            #print(os.path.basename(mix_dir))
            metadata_track = extract_metadata(track_filepaths)
            
            
            num_frames = torchaudio.info(track_filepaths[0]).num_frames
            if num_frames < self.length:
                continue
            #print(num_frames)
            offset = np.random.randint(0, num_frames - self.buffer_frames - 1)
            tracks = []
            solo_tracks = []
            stereo_info = []
            silent = True 
            metadata = []
            stereo = 0
            for tidx, track_filepath in enumerate(track_filepaths):
                # load the track
                track, sr = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.buffer_frames,
                )
                solo_track = track
                if solo_track.size()[0] == 2:
                    #print("stereo", tidx)
                    solo_track = stereo_to_mono(solo_track)
                    stereo = 1

                if track.shape[-1] != self.buffer_frames:
                    continue
                #print(os.path.basename(track_filepath))
                #
                # print(track.shape)
                instrument = metadata_track[os.path.basename(track_filepath)]
                instrument = self.instrument_ids[instrument]
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
                solo_tracks.append(solo_track)
                
                if stereo:
                    stereo_info.append(stereo)
                    metadata.append(instrument)
                    stereo = 0
                    stereo_info.append(stereo)
                    metadata.append(instrument)
                else:
                    metadata.append(instrument)
                    stereo_info.append(stereo)

            
            
            y = torch.stack(solo_tracks)
            #print(y.shape)
            y = y.sum(0)
            #print(y.shape)
            #print(type(tracks))

            if (y**2).mean() > 1e-3:
                    silent = False
                
            if silent:
                continue

        
            else:
                # loudness normalization
                track_lufs_db = self.meter.integrated_loudness(y.permute(1, 0).numpy())

                if track_lufs_db == float("-inf"):
                    continue

                delta_lufs_db = torch.tensor(
                    [self.target_track_lufs_db - track_lufs_db]
                ).float()
                gain_lin = 10.0 ** (delta_lufs_db.clamp(-120, 40.0) / 20.0)
                tracks = [gain_lin * track for track in tracks]

                nbytes = track.element_size() * track.nelement()
                nbytes_loaded += nbytes

            

            # store this example
            example ={
                "songname" : os.path.basename(mix_dir),
                "num_frames": num_frames,
                "track_filepaths": track_filepaths,
                "tracks": tracks,
                "metadata": metadata,
                "stereo": stereo_info
            }
            #print(example)

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
        #print("example_idx",example_idx)
        example = self.examples[example_idx]
        #print(example["songname"])
        # read tracks from RAM
        tracks = torch.zeros(self.max_tracks, self.length)
        instrument_id = example["metadata"]
        stereo_info = example["stereo"]

        
        track_idx = 0
        
        num_silent_tracks = self.max_tracks-len(instrument_id)
        for i in range(num_silent_tracks):
            instrument_id.append(0)
            i += 1
        
        
        
        for track in example["tracks"]:
            #print(track.shape)
            # print(len(instrument_id))
            for ch_idx in range(track.shape[0]):
                #print(ch_idx)
                if track_idx == self.max_tracks:
                    break
                else:
                    #print("track_idx",track_idx)
                    tracks[track_idx, :] = track[ch_idx, :]
                    track_idx += 1
        # print(tracks.shape)
        #print(tracks.shape)
        if len(stereo_info) != len(instrument_id):
            for i in range(len(instrument_id)-len(stereo_info)):
                stereo_info.append(0)

        instrument_id = instrument_id[:self.max_tracks]
        instrument_id = torch.tensor(instrument_id)
        stereo_info = stereo_info[:self.max_tracks]
        stereo_info = torch.tensor(stereo_info).reshape(instrument_id.shape)
        
       
        return tracks, instrument_id, stereo_info


class CambridgeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dirs: List[str],
        length: int,
        num_workers: int = 4,
        batch_size: int = 16,
        train_buffer_size_gb: float = 2.0,
        val_buffer_size_gb: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):

        if stage == "fit":
            self.train_dataset = CambridgeDataset(
                root_dirs=self.hparams.root_dirs,
                indices=[0,150],
                length=self.hparams.length,
                buffer_size_gb=self.hparams.train_buffer_size_gb,
                num_examples_per_epoch=10000,
            )

        if stage == "validate" or stage == "fit":
            self.val_dataset = CambridgeDataset(
                root_dirs=self.hparams.root_dirs,
                indices=[150,200],
                length=self.hparams.length,
                buffer_size_gb=self.hparams.val_buffer_size_gb,
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
            num_workers=4,
        )

if __name__ == "__main__":
    dataset = CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"],indices=[0,150],buffer_size_gb =4.0)
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, drop_last = True, num_workers=4)

    for i, (track, instrument_id, stereo) in enumerate(dataloader):
       print(i)
