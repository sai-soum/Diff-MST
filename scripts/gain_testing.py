
# run pretrained models over evaluation set to generate audio examples for the listening test
import os
import torch
import torchaudio
import pyloudnorm as pyln
from mst.utils import load_diffmst, run_diffmst
from mst.loss import compute_barkspectrum, compute_rms, compute_crest_factor, compute_stereo_width, compute_stereo_imbalance, AudioFeatureLoss
import json
import numpy as np
import csv
import glob
import yaml


def equal_loudness_mix(tracks: torch.Tensor, *args, **kwargs):

    meter = pyln.Meter(44100)
    target_lufs_db = -48.0

    norm_tracks = []
    for track_idx in range(tracks.shape[1]):
        track = tracks[:, track_idx : track_idx + 1, :]
        lufs_db = meter.integrated_loudness(track.squeeze(0).permute(1, 0).numpy())

        if lufs_db < -80.0:
            print(f"Skipping track {track_idx} with {lufs_db:.2f} LUFS.")
            continue

        lufs_delta_db = target_lufs_db - lufs_db
        track *= 10 ** (lufs_delta_db / 20)
        norm_tracks.append(track)

    norm_tracks = torch.cat(norm_tracks, dim=1)
    # create a sum mix with equal loudness
    sum_mix = torch.sum(norm_tracks, dim=1, keepdim=True).repeat(1, 2, 1)
    sum_mix /= sum_mix.abs().max()

    return sum_mix, None, None, None



if __name__ == "__main__":
    meter = pyln.Meter(44100)
    target_mix_lufs_db = -16.0
    target_track_lufs_db = -48.0
    output_dir = "outputs/gain_testing_diff_song_individual_tracks"
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "diffmst-16": {
            "model": load_diffmst(
                "/Users/svanka/Downloads/b4naquji/config.yaml",
                "/Users/svanka/Downloads/b4naquji/checkpoints/epoch=191-step=626608.ckpt",
            ),
            "func": run_diffmst,
        },
        # "sum": {
        #     "model": (None, None),
        #     "func": equal_loudness_mix,
        # },
    }

    ref_dir = "/Users/svanka/Downloads/DSD100subset/sources/Dev/055 - Angels In Amplifiers - I'm Alright"
    #mix_dir = "/Users/svanka/Downloads/DSD100subset/sources/Dev/055 - Angels In Amplifiers - I'm Alright"
    mix_dir = "/Users/svanka/Downloads/DSD100subset/Sources/Test/049 - Young Griffo - Facade"

    ref_tracks = glob.glob(os.path.join(ref_dir, "*.wav"))
    mix_tracks = glob.glob(os.path.join(mix_dir, "*.wav"))

    print(len(ref_tracks), len(mix_tracks))
    #order the tracks in ref_tracks to vocals, bass, other, drums
    ref_tracks_ordered = [""] * 4
    for track in ref_tracks:
        if "vocals" in track:
             ref_tracks_ordered[0] = track
        elif "bass" in track:
            ref_tracks_ordered[1] = track
        elif "other" in track:
            ref_tracks_ordered[2] = track
        elif "drums" in track:
            ref_tracks_ordered[3] = track
    ref_tracks = ref_tracks_ordered
        
    print(ref_tracks)
    # print(mix_tracks)
  

    #we will predict a mix for one track from reference, sum of two, sum of three, sum of four tracks from reference as the reference for model
    # and the mix as the input

    tracks = []
    #info = torchaudio.info(mix_tracks[0])
    
    
    track_instrument = []   
    for track in mix_tracks:
        #audio, sr = torchaudio.load(track, frame_offset = int((info.num_frames)/2 - 220500), num_frames = 441000, backend="soundfile")
        audio, sr = torchaudio.load(track,num_frames = 441000, backend="soundfile")
        if sr != 44100:
                audio = torchaudio.functional.resample(audio, sr, 44100)
        
        if audio.shape[0] == 2:
                    audio = audio.mean(dim=0, keepdim=True)
       
        tracks.append(audio)
        track_instrument.append(os.path.basename(track).replace(".wav", ""))
    

    tracks = torch.cat(tracks, dim=0)
    print("tracks shape", tracks.shape)
    tracks = tracks.unsqueeze(0)
    print("tracks shape", tracks.shape)
    
    #create a sum mix
    sum_mix, _, _, _ = equal_loudness_mix(tracks)
    print("sum_mix shape", sum_mix.shape)
    save_path = os.path.join(output_dir, f"{os.path.basename(mix_dir)}-sum_mix.wav")
    torchaudio.save(save_path, sum_mix.view(2, -1), 44100)

    ref_mix_tracks = []
    info = torchaudio.info(ref_tracks[0])
    name = "ref_mix-16="
    data = {}
    data["track_instrument"] = track_instrument
    for i , ref_track in enumerate(ref_tracks):
        instrument =  name + "-" + os.path.basename(ref_track).replace(".wav", "") 
        print(instrument)
        #name = instrument
        ref_audio, sr = torchaudio.load(ref_track, frame_offset = int((info.num_frames)/2 - 220500), num_frames = 441000, backend="soundfile")
        if sr != 44100:
                ref_audio = torchaudio.functional.resample(ref_audio, sr, 44100)
        
        #loudness normalize the reference mix to -48 LUFS
        ref_lufs_db = meter.integrated_loudness(ref_audio.squeeze().permute(1, 0).numpy())
        lufs_delta_db = target_track_lufs_db - ref_lufs_db
        ref_audio = ref_audio * 10 ** (lufs_delta_db / 20)

        #ref_mix_tracks.append(ref_audio)
        ref_mix_tracks = [ref_audio]
        ref_mix = torch.cat(ref_mix_tracks, dim=0)
        #create a stereo sum mix
        ref_mix = ref_mix.sum(dim=0, keepdim=True).repeat(1, 2, 1)
        #normalise to -16 LUFS
        ref_mix_lufs_db = meter.integrated_loudness(ref_mix.squeeze().permute(1, 0).numpy())
        lufs_delta_db = target_mix_lufs_db - ref_mix_lufs_db
        ref_mix = ref_mix * 10 ** (lufs_delta_db / 20)
        ref_save_path = os.path.join(output_dir, f"{os.path.basename(ref_dir)}-{instrument}.wav")
        torchaudio.save(ref_save_path, ref_mix.view(2, -1), 44100)

        yaml_path = os.path.join(output_dir, f"{os.path.basename(ref_dir)}-{instrument}.yaml")
        data["ref_mix"] = ref_save_path
        data["ref_instruments"] = instrument
        data["sum_mix"] = save_path
        #check if the json file exists
        print("tracks shape", tracks.shape)
        print("ref_mix shape", ref_mix.shape)



        
        for method_name, method in methods.items():
            model, mix_console = method["model"]
            func = method["func"]
            with torch.no_grad():
                result = func(
                    tracks.clone(),
                    ref_mix.clone(),
                    model,
                    mix_console,
                    track_start_idx=0,
                    ref_start_idx=0,
                )

                (
                    pred_mix,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = result

              
            bs, chs, seq_len = pred_mix.shape
            print("pred_mix shape", pred_mix.shape)
            # loudness normalize the output mix
            mix_lufs_db = meter.integrated_loudness(
                pred_mix.squeeze(0).permute(1, 0).numpy()
            )
            print("pred_mix_lufs_db", mix_lufs_db)
            #print(mix_lufs_db)
            lufs_delta_db = target_mix_lufs_db - mix_lufs_db
            pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)
            pred_mix_name = os.path.basename(mix_dir) + f"-pred_mix-ref_mix-16={instrument}.wav"
            mix_filepath =  os.path.join(output_dir, pred_mix_name)
            torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)
            # append to the json file param_dicts
            
            #print(pred_track_param_dict["input_gain"])
            
            data["pred_mix"] = pred_mix_name
            data["gain_values"] = pred_track_param_dict['input_fader']['gain_db'].detach().cpu().numpy().tolist()[0]
            #print(type(pred_track_param_dict['input_fader']['gain_db']))

            
            with open(yaml_path, "w") as f:
                yaml.dump(data, f)

      











