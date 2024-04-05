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

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    meter = pyln.Meter(44100)
    target_lufs_db = -22.0
    output_dir = "outputs/ablation"
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "diffmst-16": {
            "model": load_diffmst(
                "/Users/svanka/Downloads/b4naquji/config.yaml",
                "/Users/svanka/Downloads/b4naquji/checkpoints/epoch=191-step=626608.ckpt",
            ),
            "func": run_diffmst,
        },
        "sum": {
            "model": (None, None),
            "func": equal_loudness_mix,
        },
    }

    # get the validation examples
    examples = {
        "ecstasy": {
            "tracks": "/Users/svanka/Downloads//diffmst-examples/song1/BenFlowers_Ecstasy_Full/",
            "ref": "/Users/svanka/Codes/Diff-MST/outputs/ablation_ref_examples/_Feel it all Around_ by Washed Out (Portlandia Theme)_01/",
        },
        "by-my-side": {
            "tracks": "/Users/svanka/Downloads//diffmst-examples/song2/Kat Wright_By My Side/",
            "ref": "/Users/svanka/Codes/Diff-MST/outputs/ablation_ref_examples/The Dip - Paddle To The Stars (Lyric Video)_01/",
        },
        "haunted-aged": {
            "tracks": "/Users/svanka/Downloads//diffmst-examples/song3/Titanium_HauntedAge_Full/",
            "ref": "/Users/svanka/Codes/Diff-MST/outputs/ablation_ref_examples/Architects - _Doomsday__01/",
        },
    }


    loss = AudioFeatureLoss([0.1,0.001,1.0,1.0,0.1], 44100)
    AF = {}
    #initialise to negative infinity
   
    for example_name, example in examples.items():
       

        AF[example_name] = {}
        print(example_name)
        example_dir = os.path.join(output_dir, example_name)
        os.makedirs(example_dir, exist_ok=True)
        json_dir = os.path.join(output_dir, "AF")
        if not os.path.exists(json_dir):
            os.makedirs(json_dir, exist_ok=True)
        csv_path = os.path.join(json_dir,f"{example_name}.csv")
        # if not os.path.exists(csv_path):
        #     os.makedirs(csv_path)
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["method", "audio_type","ablation","start_idx", "stop_idx", "rms", "crest_factor", "stereo_width", "stereo_imbalance", "barkspectrum", "net_AF_loss"])
            f.close()
        ref_loudness_target = -16.0
        
        # --------------first find all the tracks----------------
        track_filepaths = []
        for root, dirs, files in os.walk(example["tracks"]):
            for filepath in files:
                if filepath.endswith(".wav"):
                    track_filepaths.append(os.path.join(root, filepath))

        print(f"Found {len(track_filepaths)} tracks.")

        # ----------------load the tracks----------------------------
        tracks = []
        lengths = []
        for track_idx, track_filepath in enumerate(track_filepaths):
            audio, sr = torchaudio.load(track_filepath, backend="soundfile")

            if sr != 44100:
                audio = torchaudio.functional.resample(audio, sr, 44100)

            # loudness normalize the tracks to -48 LUFS
            lufs_db = meter.integrated_loudness(audio.permute(1, 0).numpy())
            # lufs_delta_db = -48 - lufs_db
            # audio = audio * 10 ** (lufs_delta_db / 20)

            print(track_idx, os.path.basename(track_filepath), audio.shape, sr, lufs_db)

            if audio.shape[0] == 2:
                audio = audio.mean(dim=0, keepdim=True)

            chs, seq_len = audio.shape

            for ch_idx in range(chs):
                tracks.append(audio[ch_idx : ch_idx + 1, :])
                lengths.append(audio.shape[-1])

        # find max length and pad if shorter
        max_length = max(lengths)
        min_length = min(lengths)   
        for track_idx in range(len(tracks)):
            tracks[track_idx] = torch.nn.functional.pad(
                tracks[track_idx], (0, max_length - lengths[track_idx])
            )

        # stack into a tensor
        tracks = torch.cat(tracks, dim=0)
        tracks = tracks.view(1, -1, max_length)
        tracks_length = max_length
        refs = glob.glob(os.path.join(example["ref"],"*.wav"))
        print("found refs", len(refs))
        for ref in refs:
            ref_name = os.path.basename(ref).replace(".wav", "")
            test_type = ref_name.split("_")[-2] + "_" + ref_name.split("_")[-1]
            print(test_type)
            
            print(ref_name)
            AF[example_name]["ref"] = {}
            AF[example_name]["pred_mix"] = {}
            ref_audio, ref_sr = torchaudio.load(ref, backend="soundfile")
            if ref_sr != 44100:
                ref_audio = torchaudio.functional.resample(ref_audio, ref_sr, 44100)
            print(ref_audio.shape, ref_sr)
            ref_length = ref_audio.shape[-1]
            ref_audio = ref_audio.view(1, 2, -1)

            #loudness normalize the reference mix to -16 LUFS
            ref_lufs_db = meter.integrated_loudness(ref_audio.squeeze().permute(1, 0).numpy())
            lufs_delta_db = ref_loudness_target - ref_lufs_db
            ref_audio = ref_audio * 10 ** (lufs_delta_db / 20)


            # --------------run inference----------------
            #print(tracks.shape)
            track_idx = int(tracks_length / 2)
            ref_idx = int(ref_length / 2)
            mix_tracks = tracks[..., track_idx - 220500 : track_idx + 220500]
            ref_analysis = ref_audio[..., ref_idx - 220500 : ref_idx + 220500]

            ref_path = os.path.join(example_dir, os.path.basename(ref).replace(".wav", "-ref-16.wav"))
            torchaudio.save(ref_path, ref_analysis.squeeze(), 44100)

            for method_name, method in methods.items():
                AF[example_name]["ref"] [method_name] = {}
                AF[example_name]["pred_mix"] [method_name] = {}

                print(method_name)
                model, mix_console = method["model"]
                func = method["func"]

                with torch.no_grad():
                    result = func(
                        mix_tracks.clone(),
                        ref_analysis.clone(),
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
                lufs_delta_db = target_lufs_db - mix_lufs_db
                pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)
                name = os.path.basename(ref).replace(".wav", "-pred_mix-16.wav")
                mix_filepath =  os.path.join(example_dir, f"{method_name}_{name}")
                torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

                # compute audio features
               
                AF[example_name]["pred_mix"][method_name]["mix-rms"] = 0.1*compute_rms(pred_mix, sample_rate = sr).mean().detach().cpu().numpy()
                AF[example_name]["pred_mix"][method_name]["mix-crest_factor"] = 0.001*compute_crest_factor(pred_mix, sample_rate = sr).mean().detach().cpu().numpy()
                AF[example_name]["pred_mix"][method_name]["mix-stereo_width"] = 1.0*compute_stereo_width(pred_mix, sample_rate = sr).detach().cpu().numpy()
                AF[example_name]["pred_mix"][method_name]["mix-stereo_imbalance"] = 1.0*compute_stereo_imbalance(pred_mix, sample_rate = sr).detach().cpu().numpy()
                AF[example_name]["pred_mix"][method_name]["mix-barkspectrum"] = 0.1*compute_barkspectrum(pred_mix, sample_rate = sr).mean().detach().cpu().numpy()
                
                AF[example_name]["ref"][method_name]["mix-rms"] = 0.1*compute_rms(ref_analysis, sample_rate = sr).mean().detach().cpu().numpy()
                AF[example_name]["ref"][method_name]["mix-crest_factor"] = 0.001*compute_crest_factor(ref_analysis, sample_rate = sr).mean().detach().cpu().numpy()
                AF[example_name]["ref"][method_name]["mix-stereo_width"] = 1.0*compute_stereo_width(ref_analysis, sample_rate = sr).detach().cpu().numpy()
                AF[example_name]["ref"][method_name]["mix-stereo_imbalance"] = 1.0*compute_stereo_imbalance(ref_analysis, sample_rate = sr).detach().cpu().numpy()
                AF[example_name]["ref"][method_name]["mix-barkspectrum"] = 0.1*compute_barkspectrum(ref_analysis, sample_rate = sr).mean().detach().cpu().numpy()

                AF_loss = loss(pred_mix, ref_analysis)
                AF[example_name]["pred_mix"][method_name]["net_AF_loss"]  = sum(AF_loss.values()).detach().cpu().numpy()
                AF[example_name]["ref"][method_name]["net_AF_loss"] = AF[example_name]["pred_mix"][method_name]["net_AF_loss"] 


                # save resulting audio and parameters
                #append to csv the method name, audio section, audio features values and net loss on different columns
                with open(csv_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([method_name, "pred_mix", test_type,  track_idx - 220500, track_idx + 220500,AF[example_name]["pred_mix"][method_name]["mix-rms"], AF[example_name]["pred_mix"][method_name]["mix-crest_factor"], AF[example_name]["pred_mix"][method_name]["mix-stereo_width"], AF[example_name]["pred_mix"][method_name]["mix-stereo_imbalance"], AF[example_name]["pred_mix"][method_name]["mix-barkspectrum"], AF[example_name]["pred_mix"][method_name]["net_AF_loss"]])
                    writer.writerow([method_name, "ref",  test_type, ref_idx - 220500, ref_idx + 220500,AF[example_name]["ref"][method_name]["mix-rms"], AF[example_name]["ref"][method_name]["mix-crest_factor"], AF[example_name]["ref"][method_name]["mix-stereo_width"], AF[example_name]["ref"][method_name]["mix-stereo_imbalance"], AF[example_name]["ref"][method_name]["mix-barkspectrum"], AF[example_name]["ref"][method_name]["net_AF_loss"]])
                    f.close()
        
        
   
    #write disctionary to json


