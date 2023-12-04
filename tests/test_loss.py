import torch
import torchaudio

from mst.loss import AudioFeatureLoss, ParameterEstimatorLoss
from mst.loss import (
    compute_crest_factor,
    compute_melspectrum,
    compute_barkspectrum,
    compute_rms,
    compute_stereo_imbalance,
    compute_stereo_width,
)

transforms = [
    compute_rms,
    compute_crest_factor,
    compute_stereo_width,
    compute_stereo_imbalance,
    compute_barkspectrum,
]
weights = [10.0, 0.1, 10.0, 100.0, 0.1]

sample_rate = 44100

# loss = AudioFeatureLoss(weights, sample_rate, stem_separation=False)

ckpt_path = "/import/c4dm-datasets-ext/Diff-MST/DiffMST-Param/0ymfi1pp/checkpoints/epoch=5-step=10842.ckpt"
loss = ParameterEstimatorLoss(ckpt_path)

# test with audio examples
input, _ = torchaudio.load("outputs/output/pred_mix.wav")
target, _ = torchaudio.load("outputs/output/ref_mix.wav")


input = input.unsqueeze(0)
target = target.unsqueeze(0)

input = input.repeat(4, 1, 1)
target = target.repeat(4, 1, 1)

# input[0, ...] = 0.0001 * torch.randn_like(input[0, ...])
# target[0, ...] = 0.0001 * torch.randn_like(input[0, ...])


loss_val = loss(input, target)

print(loss_val)
