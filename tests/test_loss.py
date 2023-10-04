import torch
import torchaudio

from mst.loss import AudioFeatureLoss
from mst.loss import (
    compute_crest_factor,
    compute_melspectrum,
    compute_rms,
    compute_stereo_imbalance,
    compute_stereo_width,
)

transforms = [
    compute_rms,
    compute_crest_factor,
    compute_stereo_width,
    compute_stereo_imbalance,
    compute_melspectrum,
]
weights = [10.0, 0.1, 10.0, 100.0, 0.1]

sample_rate = 44100

loss = AudioFeatureLoss(weights, sample_rate)

# test with audio examples
input, _ = torchaudio.load("output/sum_mix.wav")
target, _ = torchaudio.load("output/ref_mix.wav")

input = input.repeat(2, 1)
input = input.unsqueeze(0)
target = target.unsqueeze(0)

print(input.shape, target.shape)

loss_val = loss(input, target)

print(loss_val)
