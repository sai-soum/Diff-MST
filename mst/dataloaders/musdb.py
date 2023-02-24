import os
import glob
import os
import random
from typing import List

import numpy as np
import torch
import torchaudio
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import random


# from tensorboard_log import tb_audio_log

from torch.utils.tensorboard import SummaryWriter

# from train import mix_gen
# from dataset_generation import load_tracks,


"""
Class to convert Stereo Audio to Mono
"""


class ToMono:
    def __init__(self, channel_first=True):
        self.channel_first = channel_first

    def __call__(self, x):
        assert len(x.shape) == 2, "Can only take two dimenshional Audio Tensors"

        output = (
            torch.mean(x, dim=0, keepdim=True)
            if self.channel_first
            else torch.mean(x, dim=1, keepdim=True)
        )
        return output


def save_audio(audio: torch.Tensor, path: str, sample_rate: int = 44100):
    # print(audio.size())
    path_to_save = os.path.join("/mix_audios/", path)
    # print(f"Saving audio to {path_to_save}")
    # os.umask(0)
    # os.makedirs(path_to_save, exist_ok=True)
    audio = audio.squeeze(0).cpu()

    # torchaudio.save(path, audio, sample_rate)


"""
params: [nb,nstems,2]
ch_data: [nb,nstems,nch_per_stem, nsamples]
"""


def mix_gen(
    stems: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    """

    Args:
        stems (torch.Tensor): Batch of stems with shape (bs, nstems, nchs, nsamples)
        params (torch.Tensor): Batch of mix parameters with shape (bs, nstems, nparams)

    Returns:
        mix (torch.Tensor): Batch of mixes with shape (bs, 2, nsamples)
    """
    bs, nstems, nch_per_stem, nsamples = stems.size()

    # ---- gain ----
    #print("params shape", params.shape)
    gain_dB = params[...,0].view(bs, nstems,1)
    #print ("gain_dB", gain_dB.shape)
    #print("gain_dB", gain_dB)
    #print("gain_db:",gain_dB.shape)
    gain_dB = gain_dB * 12
    #print("gain_dB", gain_dB)
    gain_lin = 10 ** (gain_dB / 20.0)
    #print("gain_lin", gain_lin)
    #print("gain_lin", gain_lin.shape)
    #print("ch_data", ch_data.shape)
    stems_with_gain = stems * gain_lin[...,None]
    #print("stems_with_gain", stems_with_gain.shape)

    # ---- pan ----
    pan = params[...,1]
    #print("pan", pan.shape)
    #print(pan)
    pan_theta = pan*torch.pi/2

    cos_weight = torch.cos(pan_theta)
    sin_weight = torch.sin(pan_theta)
    #print("cos_weight", cos_weight.shape)
    #print("sin_weight", sin_weight.shape)
    #print("stems_with_gain", stems_with_gain.shape)
    pan_params = torch.stack([cos_weight,sin_weight],dim=-1).view(bs, nstems, 2, 1)
    stems_with_mix_and_pan_weight = stems_with_gain * pan_params
    #print("stems_with_mix_and_pan_weight", stems_with_mix_and_pan_weight.shape)
    # ---- mix ----
    mix = torch.sum(stems_with_mix_and_pan_weight,dim=1)
    #print("mix", mix.shape)

    return mix


"""
stems = [nb,nstems,nch_per_stem, nsamples]
params = [nb,nstems,1]
"""


def mix_gen_gain(
    stems: torch.Tensor, params: torch.Tensor, songname: str = ""
) -> torch.Tensor:

    gain_linear_params = 10 ** (params * 12 / 20.0)
    gain_linear_params = gain_linear_params.unsqueeze(-1)
    # print("stems",stems.shape)
    # print("params",gain_linear_params.shape)
    stems_with_gain = gain_linear_params * stems
    mix = torch.sum(stems_with_gain, dim=1)
    # ßprint(f"mix: {mix.size()}")
    # save_audio(mix, f"{songname[:5]}.wav")
    return mix


#implement indices for the split


class MUSEDB_on_the_fly(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = "/import/c4dm-datasets/MUSDB18HQ",
        sample_rate: int = 44100,
        duration: int = 5,
        num_tracks: int = [5,5],
        type: str = "train",
        experiment_type: str = "gain_pan",
        diff_sections: bool = True,
        
       ) -> None:

        super().__init__()
        # navugate to the train/ test folder. MUSEDB has no val folder. So, we will break the train folder into val and train
        if type == "train":
            self.data_dir = os.path.join(data_dir, "train")
            self.indices = [0,80]
        elif type == "test":
            self.data_dir = os.path.join(data_dir, "test")
            self.indices = [0,50]
        elif type == "val":
            self.data_dir = os.path.join(data_dir, "train")
            self.indices = [80,100]
        else:
            raise ValueError("type must be either 'train' or 'test'")

        self.sample_rate = sample_rate
        self.duration = duration
        self.experiment_type = experiment_type
        self.num_tracks = num_tracks
        self.diff_sections = diff_sections
        #self.num_examples_per_epoch = num_examples_per_epoch
       
        # find all songs in the data directory at MUSEDB18/train or MUSEDB18/test
        self.songdirs = glob.glob(os.path.join(self.data_dir, "*"))
        self.songdirs = sorted(self.songdirs)
        
        self.songdirs = self.songdirs[self.indices[0]:self.indices[1]]
        print(f"Found {len(self.songdirs)} songs in {self.data_dir}.")

    def __len__(self):
        return len(self.songdirs)

    def __getitem__(self, _):
        # select a song directory at random
        songdir = np.random.choice(self.songdirs)
        # songdir = self.songdirs[0]
        songname = os.path.basename(songdir)
        #print(songname)
        #print(type(songname))
        filepaths = glob.glob(os.path.join(songdir,"*.wav"))
        #print(filepaths)
        stereo_to_mono = ToMono()
        #get all the data
        
        
        stems = []
        
        for filepath in filepaths:
            if filepath.endswith(".wav"):
                md = torchaudio.info(filepath)
                #need to find a way to generate random start index with 
                #print(md)

                #start_idx = torch.randint(0, md.num_frames - self.sample_rate*self.duration - 1, (1,))
                start_idx_stems = int((md.num_frames/2) - (self.sample_rate*self.duration - 1)/2)
                start_idx_stems = int(start_idx_stems)
                #print(f'start_idx_stems: {start_idx_stems}')
                #print(start_idx)
                if self.diff_sections:
                    start_idx_mix = torch.randint(0, int(md.num_frames - self.sample_rate*self.duration - 1), (1,))
                    start_idx_mix = int(start_idx_mix)
                    #print(f'start_idx_mix: {start_idx_mix}')
                else:
                    start_idx_mix = start_idx_stems
                break

        
        
        ctr = 0
        for filepath in filepaths:
            if  not filepath.endswith("mixture.wav") and not filepath.endswith("accompaniment.wav"):
                stem, sr = torchaudio.load(filepath, frame_offset= start_idx_stems, num_frames=int(self.duration*self.sample_rate))
                #print(filepath)
                #print(f"{filepath}_stem shape",stem.shape)
                if torch.isnan(stem).any():
                    #print(stem)
                    raise ValueError(f"nan in {filepath}")

                if sr != self.sample_rate:
                    stem = torchaudio.transforms.Resample(sr, self.sample_rate)(stem)
                    # if torch.isnan(stem).any():
                    #     raise ValueError("nan after resampling")
                if stem.size()[0] == 2:
                    stem = stereo_to_mono(stem)
                    # if torch.isnan(stem).any():
                    #     raise ValueError("nan after stereo to mono")
           

                stems.append(stem)
                ctr += 1
            if self.num_tracks <= ctr:
                break
                
        #print("its mix",mix.shape)
        #print("number of tracks", i+1)
        
        stems = torch.stack(stems)
            #print("stems: ",stems.shape)
        
        if torch.isnan(stems).any():
                raise ValueError("stems contains nan in dl")

        params = torch.rand(ctr,2)
        #params[...,1]=0.5
        #params[...,0]= 0.5
        #print(params)
        
        # params = torch.rand(ctr,2)
        # if self.experiment_type == "gain_only":
        #     params[...,1]= 0.5
            #params = torch.round(params, decimals = 2)
        
        
        params = params.unsqueeze(0)
        stems = stems.unsqueeze(0)
        #print("\n\nmixing song for dataloader") 
        orig_mix = mix_gen(stems, params)
        #we dont want to use accompaniment with other tracks as this is just sum of other, bass and drums. It might confuse the model
        if self.diff_sections:
            ctr_1 = 0
            stems_for_rmix = []
            for filepath in filepaths:
                if not filepath.endswith("mixture.wav") and not filepath.endswith("accompaniment.wav"):
                    stem, sr = torchaudio.load(filepath, frame_offset= start_idx_mix, num_frames=int(self.duration*self.sample_rate))
                    #print(filepath)
                    #print(f"{filepath}_stem shape",stem.shape)
                    if torch.isnan(stem).any():
                        #print(stem)
                        raise ValueError(f"nan in {filepath}")

                    if sr != self.sample_rate:
                        stem = torchaudio.transforms.Resample(sr, self.sample_rate)(stem)
                        # if torch.isnan(stem).any():
                        #     raise ValueError("nan after resampling")
                    if stem.size()[0] == 2:
                        stem = stereo_to_mono(stem)
                        # if torch.isnan(stem).any():
                        #     raise ValueError("nan after stereo to mono")
            

                    stems_for_rmix.append(stem)
                    ctr_1 += 1
                if self.num_tracks <= ctr_1:
                    break
            stems_for_rmix = torch.stack(stems_for_rmix)
            if torch.isnan(stems_for_rmix).any():
                raise ValueError("stems contains nan in dl")
            stems_for_rmix = stems_for_rmix.unsqueeze(0)
            ref_mix = mix_gen(stems_for_rmix, params)
        
        else:
            
            ref_mix = orig_mix

        
       
        
        #save_audio(mix, path = songname, sample_rate = self.sample_rate)
        '''
        mix shape is torch.Size([1, 2, 220500])
        stems shape is torch.Size([1, 5, 1, 220500])
        stems: bass, accomp, drums, vocals, other
        params shape is torch.Size([1, 5, 2])
        '''
        #print(f'before squeezing in dataloader: mix shape is {mix.shape},stems shape is {stems.shape}, params shape is {params.shape}')
        
        #stems = stems.squeeze(0)
        params = params.squeeze(0)
        ref_mix = ref_mix.squeeze(0)
        stems = stems.squeeze(0)
        orig_mix = orig_mix.squeeze(0)


        #print(f'after squeezing in dataloader: ref mix shape is {ref_mix.shape}, orig mix shape is {orig_mix.shape},stems shape is {stems.shape}, params shape is {params.shape}')
        # if torch.isnan(stems).any():
        #     raise ValueError("stems contains nan returning from datal;oader")
        return songname, stems, ref_mix, orig_mix,
        
        #original mix: the mix of the stems that go into the encoder/model = the idea would be to compute loss against this one 
        # because gain and pan values are static right now but we just want to see if the model can be invariant to the content in some sense
        #reference mix(also called as mix here): the mix of the stems that go into the encoder/model





