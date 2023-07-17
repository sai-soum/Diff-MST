import os
import torch
import argparse
import torchaudio
import pytorch_lightning as pl

from mst.system import System
from mst.utils import load_model

# load a pretrained model and create a mix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to config.yaml for pretrained model checkpoint.",
    )
    parser.add_argument(
        "ckpt_path",
        type=str,
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument(
        "track_dir",
        type=str,
        help="Path to directory containing tracks.",
    )
    parser.add_argument(
        "ref_mix",
        type=str,
        help="Path to reference mix.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory.",
        default="output",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Whether to use GPU.",
    )
    args = parser.parse_args()

    # load model
    model = load_model(
        args.config_path,
        args.ckpt_path,
        map_location="gpu" if args.use_gpu else "cpu",
    )
    sample_rate = 44100
    print(f"Loaded model: {os.path.basename(args.ckpt_path)}\n")

    # load multitracks (wav files only)
    track_paths = [
        os.path.join(args.track_dir, f)
        for f in os.listdir(args.track_dir)
        if ".wav" in f
    ]

    tracks = []
    print("Loading tracks...")
    for idx, track_path in enumerate(track_paths):
        track, track_sr = torchaudio.load(track_path)
        if track_sr != sample_rate:
            ref_mix = torchaudio.functional.resample(track, track_sr, sample_rate)
        tracks.append(track)
        print(f"({idx+1}/{len(track_paths)}): {os.path.basename(track_path)}")

    tracks = torch.cat(tracks, dim=0)
    tracks = tracks[:, 524288 : 2 * 524288]

    # load reference track
    ref_mix, ref_sr = torchaudio.load(args.ref_mix)
    if ref_sr != sample_rate:
        ref_mix = torchaudio.functional.resample(ref_mix, ref_sr, sample_rate)
    ref_mix = ref_mix[:, :524288]
    print(f"\nLoaded reference mix: {os.path.basename(args.ref_mix)}.")
    print(f"tracks: {tracks.shape}  ref_mix: {ref_mix.shape}\n")

    # create sum mix
    sum_mix = torch.sum(tracks, dim=0, keepdim=True)

    # create mix with model
    with torch.no_grad():
        result = model(tracks.unsqueeze(0), ref_mix.unsqueeze(0))
        (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        ) = result
        mix = mix.squeeze(0)

    # save mix
    os.makedirs(args.output_dir, exist_ok=True)
    torchaudio.save(os.path.join(args.output_dir, "pred_mix.wav"), mix, sample_rate)
    torchaudio.save(os.path.join(args.output_dir, "ref_mix.wav"), ref_mix, sample_rate)
    torchaudio.save(os.path.join(args.output_dir, "sum_mix.wav"), sum_mix, sample_rate)
    print(f"Saved mixes to {args.output_dir}.\n")
