import torch
import torchaudio
import matplotlib.pyplot as plt

from dasp_pytorch.functional import noise_shaped_reverberation


bs = 2
chs = 2
seq_len = 262144
sample_rate = 44100

# x = torch.randn(bs, chs, seq_len)
# x = x / x.abs().max().clamp(min=1e-8)
# x *= 10 ** (-24 / 20)

x, sr = torchaudio.load("tests/target-gtr.wav")
x = x.repeat(2, 1)
x = x.unsqueeze(0)
print(x.shape)

band_gains = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
band_decays = torch.tensor(
    [0.8, 0.7, 0.8, 0.6, 0.6, 0.7, 0.8, 0.8, 0.99, 0.8, 0.9, 1.0]
)

band0_gain = torch.rand(1)
band1_gain = torch.rand(1)
band2_gain = torch.rand(1)
band3_gain = torch.rand(1)
band4_gain = torch.rand(1)
band5_gain = torch.rand(1)
band6_gain = torch.rand(1)
band7_gain = torch.rand(1)
band8_gain = torch.rand(1)
band9_gain = torch.rand(1)
band10_gain = torch.rand(1)
band11_gain = torch.rand(1)
band0_decay = torch.rand(1)
band1_decay = torch.rand(1)
band2_decay = torch.rand(1)
band3_decay = torch.rand(1)
band4_decay = torch.rand(1)
band5_decay = torch.rand(1)
band6_decay = torch.rand(1)
band7_decay = torch.rand(1)
band8_decay = torch.rand(1)
band9_decay = torch.rand(1)
band10_decay = torch.rand(1)
band11_decay = torch.rand(1)

mix = torch.tensor([0.05])

y = noise_shaped_reverberation(
    x,
    sample_rate,
    band0_gain,
    band1_gain,
    band2_gain,
    band3_gain,
    band4_gain,
    band5_gain,
    band6_gain,
    band7_gain,
    band8_gain,
    band9_gain,
    band10_gain,
    band11_gain,
    band0_decay,
    band1_decay,
    band2_decay,
    band3_decay,
    band4_decay,
    band5_decay,
    band6_decay,
    band7_decay,
    band8_decay,
    band9_decay,
    band10_decay,
    band11_decay,
    mix,
    num_samples=88200,
    num_bandpass_taps=1023,
)

print(y.shape)
y /= y.abs().max().clamp(min=1e-8)
torchaudio.save("tests/reverb.wav", y.view(2, -1), sample_rate)
