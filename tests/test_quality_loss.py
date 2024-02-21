import torch
import torchaudio

from mst.loss import QualityLoss

sample_rate = 44100

# loss = AudioFeatureLoss(weights, sample_rate, stem_separation=False)

ckpt_path = "/import/c4dm-datasets-ext/Diff-MST/DiffMST-Param/0ymfi1pp/checkpoints/epoch=5-step=10842.ckpt"
loss = QualityLoss()

# test with audio examples
input, _ = torchaudio.load(
    "outputs/Kat Wright_By My Side-->The Dip - Paddle To The Stars (Lyric Video)/mono_mix_section.wav"
)
target, _ = torchaudio.load(
    "outputs/Kat Wright_By My Side-->The Dip - Paddle To The Stars (Lyric Video)/ref_mix_section.wav"
)


input = input.unsqueeze(0)
target = target.unsqueeze(0)

# input = input.repeat(4, 1, 1)
# target = target.repeat(4, 1, 1)

# input[0, ...] = 0.0001 * torch.randn_like(input[0, ...])
# target[0, ...] = 0.0001 * torch.randn_like(input[0, ...])

target_loss_val = loss(target)
input_loss_val = loss(input)

print(f"target loss: {target_loss_val.mean()}  input loss: {input_loss_val.mean()}")
