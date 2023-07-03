import torch
import torchaudio
import matplotlib.pyplot as plt

from dasp_pytorch.functional import reverb


bs = 2
chs = 2
seq_len = 262144
sample_rate = 44100

# x = torch.randn(bs, chs, seq_len)
# x = x / x.abs().max().clamp(min=1e-8)
# x *= 10 ** (-24 / 20)

x, sr = torchaudio.load("tests/target-gtr.wav")
x = x.unsqueeze(0)

band_gains = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
band_decays = torch.tensor(
    [0.8, 0.7, 0.8, 0.6, 0.6, 0.7, 0.8, 0.8, 0.99, 0.8, 0.9, 1.0]
)

mix = torch.tensor([0.05])

y = reverb(
    x,
    sample_rate,
    band_gains,
    band_decays,
    mix,
    num_samples=88200,
    num_bandpass_taps=1023,
)

print(y.shape)
y /= y.abs().max().clamp(min=1e-8)
torchaudio.save("tests/reverb.wav", y.view(2, -1), sample_rate)
