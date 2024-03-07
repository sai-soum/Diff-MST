# script to compare two mixes based on their features
import os
import torch
import argparse
import torchaudio
import matplotlib.pyplot as plt

from mst.loss import (
    compute_barkspectrum,
    compute_crest_factor,
    compute_rms,
    compute_stereo_imbalance,
    compute_stereo_width,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_a", type=str)
    parser.add_argument("input_b", type=str)
    parser.add_argument("--output_dir", type=str, default="outputs/compare")
    args = parser.parse_args()

    input_a_filename = os.path.basename(args.input_a).split(".")[0]
    input_b_filename = os.path.basename(args.input_b).split(".")[0]
    run_name = f"{input_a_filename}-{input_b_filename}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # load audio files
    input_a, input_a_sample_rate = torchaudio.load(args.input_a)
    input_b, input_b_sample_rate = torchaudio.load(args.input_b)

    # -------------- compute features ----------------
    a_barkspectrum = compute_barkspectrum(input_a, sample_rate=44100)
    b_barkspectrum = compute_barkspectrum(input_b, sample_rate=44100)

    a_crest_factor = compute_crest_factor(input_a)
    b_crest_factor = compute_crest_factor(input_b)

    a_rms = compute_rms(input_a)
    b_rms = compute_rms(input_b)

    a_stereo_imbalance = compute_stereo_imbalance(input_a)
    b_stereo_imbalance = compute_stereo_imbalance(input_b)

    a_stereo_width = compute_stereo_width(input_a)
    b_stereo_width = compute_stereo_width(input_b)

    # -------------- plot features ----------------

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs[0].plot(a_barkspectrum[0, :, 0], label="A-mid", color="tab:orange")
    axs[0].plot(b_barkspectrum[0, :, 0], label="B-mid", color="tab:blue")
    axs[1].plot(a_barkspectrum[0, :, 1], label="A-side", color="tab:orange")
    axs[1].plot(b_barkspectrum[0, :, 1], label="B-side", color="tab:blue")
    axs[0].legend()
    axs[1].legend()
    plt.savefig(os.path.join(output_dir, f"bark_spectrum.png"))
    plt.close("all")
