import os
import torch
import argparse
import torchaudio
import pyloudnorm as pyln
import pytorch_lightning as pl

from mst.system import System
from mst.utils import load_model

# load a pretrained model and test on random mixes

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
        "track_dirs",
        nargs="+",
        help="List of paths to directories containing tracks.",
    )
    args = parser.parse_args()

    # load model
    model = load_model(
        args.config_path,
        args.ckpt_path,
        map_location="gpu" if args.use_gpu else "cpu",
    )
    sample_rate = 44100
    meter = pyln.Meter(sample_rate)

    print(f"Loaded model: {os.path.basename(args.ckpt_path)}\n")
