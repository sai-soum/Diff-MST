from argparse import ArgumentParser
import os
import sys
import glob
import argparse
from fx import pan 
from fx import gain 
import soundfile as sf
import numpy as np
from tqdm import tqdm
import torch
import torchaudio
from mst.mix_style_transfer.mst.dataloaders.dataloader import mix_gen



def get_songname(songdir, dataset="medleydb"):
    """ Given the directory of a song extract the proper song name.

    """
    if dataset == "medleydb":
        songname = os.path.basename(songdir)
    else:
        raise ValueError(f"Invalid dataset {dataset}. Must be 'medleydb'")
    print(songname)
    return songname



def load_tracks_and_mix(songdir, mixdir, dataset="medleydb", num_track=4):
    """ Load all wav files from directory into numpy array.

    Args:
        sondir (str): Path to directory containing mulitrack files.
        mixdir (str): Path to directory containing mix files.
        dataset (str, optional): Dataset source (to define directory structure).
        num_tracks (int, optional): Number of tracks to load.
    Returns:
        ch_data (ndarray): Audio data with shape [no. of tracks, samples].
        sr (int): Sample rate of the audio data. 

    """
    #print(mixdir)
    if dataset == "medleydb":
        songname = os.path.basename(songdir)
        rawtracks = glob.glob(os.path.join(songdir, f"{songname}_RAW", "*.wav"))
    else:
        raise ValueError(f"Invalid dataset {dataset}. Must be 'medleydb'")
    max_num_samples    = 0

    total_num_tracks = 0
    stems = []

    if len(rawtracks) < num_track:
        print("Not enough tracks")
        os.rmdir(mixdir)
        return

    #Go through tracks in the songdir and load them into stems
    for track_file in rawtracks:
        #counter to load only the num of tracks we need to make a mix
        if(total_num_tracks < num_track):
            #load the track
            track_data, sr = torchaudio.load(track_file) 
            #print(track_data.shape)
            # load audio (with shape (samples, channels))
            #sample to 44.1KHz
            if sr != 44100:
                track_data = torchaudio.transforms.Resample(sr, 44100)(track_data)
            #convert to mono
            if track_data.size()[0] > 1:
                track_data = torchaudio.transforms.DownmixMono()(track_data)
            stems.append(track_data)     
            #write the stem in the mix directory
            torchaudio.save(os.path.join(mixdir, f"{songname}_{total_num_tracks}.wav"), track_data, sr)  
            #keeps a track of number of tracks 
            total_num_tracks += 1 

    stems = torch.stack(stems)
    #generate random parameter tensor, 2 for each stem: gain, pan
    params = torch.rand((num_track, 2))
    torch.save(params, os.path.join(mixdir, f"{songname}_params.pt"))
    stems = stems.unsqueeze(0)
    params = params.unsqueeze(0)
    #params = [None, params]
    #generate the mix using stems and mix 
    #print("shape of params", params.shape)
    #print("before mixing: stems shape", stems.shape)

    mix = mix_gen(stems, params)
    mix = mix.squeeze(0)
    #print(mix.shape)
    #print(type(mix))
    #save the mix in the mix directory
    torchaudio.save(os.path.join(mixdir, f"{songname}_mix.wav"), mix, sr)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputdir", help="directory pointing to multitrack files", type=str, default="/import/c4dm-datasets/MedleyDB_V1/V1")
    parser.add_argument("-d", "--dataset", help="dataset used for the mix synthesis process", type=str, default="medleydb")
    parser.add_argument("-v", "--verbose", help="increase output verbosity", default= True)
    parser.add_argument("-o", "--output", help="path to output directory for saving mixes", type=str, default="data/")
    parser.add_argument("-w", "--overwrite", help="if set exisitng mixes will be overwritten", type=bool, default=False)
    parser.add_argument("-nt", "--num_tracks", help="number of tracks to be loaded from each song", type=int, default=4)
    parser.add_argument("-ns", "--num_songs", help="number of songs to be loaded from the dataset", type=int, default=100)
    parser.add_argument("-nso", "--num_song_bool", help="take the default number of songs to load from dataset", type=bool, default=False)
    parser.add_argument("-split_tr", "--split_train", help="split the percent of dataset into train ", type=float, default=0.75)
    parser.add_argument("-split_v", "--split_val", help="split the percent of dataset into val", type=float, default=0.15)
    parser.add_argument("-split_ts", "--split_test", help="split the percent of dataset into test", type=float, default=0.1)



    args = parser.parse_args()
    output = args.output
    num_tracks = args.num_tracks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\ndevice",device)
    # locate all song directories with source tracks
    songdirs = glob.glob(os.path.join(args.inputdir, "*"))
    #print("\n" , songdirs)


    #create directory for saving output random mixes
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
        if not os.path.isdir(os.path.join(args.output, "train")):
            os.makedirs(os.path.join(args.output, "train"))
        if not os.path.isdir(os.path.join(args.output, "test")):
            os.makedirs(os.path.join(args.output, "test"))
        if not os.path.isdir(os.path.join(args.output, "val")):
            os.makedirs(os.path.join(args.output, "val"))

    if args.verbose:
        print(f"Discovered {len(songdirs)} directories with mutltitracks.")
    
    if args.split_train + args.split_val + args.split_test != 1.0:
        raise ValueError("Train, val, and test splits must sum to 1.0")


    if args.num_song_bool:
        num_songs_to_process = args.num_songs
        train_data = args.split_train * num_songs_to_process
        val_data = args.split_val * num_songs_to_process
        test_data = args.split_test * num_songs_to_process
    else:
        num_songs_to_process = len(songdirs)
        train_data = args.split_train * num_songs_to_process
        val_data = args.split_val * num_songs_to_process
        test_data = args.split_test * num_songs_to_process

    # iterate over song directories creating mix of each song
    train_songs = 0
    val_songs = 0
    for sidx, songdir in enumerate(songdirs):
        if(sidx < num_songs_to_process):
            #print(songdir)
            print(f"* Song {sidx+1:3d}/{num_songs_to_process:3d}")
        
    
            songname = get_songname(songdir, dataset=args.dataset)  
                    # extract the song name
            if train_songs < train_data:
                mixdir = os.path.join(args.output,"train", songname)
                train_songs += 1
            else:
                if val_songs < val_data:
                    mixdir = os.path.join(args.output,"val", songname)
                    val_songs += 1
                else:
                    mixdir = os.path.join(args.output,"test", songname)

            # directory to store all random mixes of this song

            if not os.path.isdir(mixdir):
                os.makedirs(mixdir) 
                #print("dir created")   # create mix directory if it doesn't exist
            else:
                if not args.overwrite: # if directory exisits move onto the next song 
                    print("dir exists")
                    
            
            #loads, saves and mixes the tracks
            load_tracks_and_mix(songdir, mixdir, dataset=args.dataset, num_track = args.num_tracks) #dimension is (no. of tracks = 5 , no. of samples 
       

        

