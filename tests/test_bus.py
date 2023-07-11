import torch
import torchaudio
import matplotlib.pyplot as plt

from dasp_pytorch.functional import stereo_bus


bs = 2
chs = 2
seq_len = 262144
sample_rate = 44100

# x = torch.randn(bs, chs, seq_len)
# x = x / x.abs().max().clamp(min=1e-8)
# x *= 10 ** (-24 / 20)

x, sr = torchaudio.load("tests/target-gtr.wav")
x = x.unsqueeze(0)

x = torch.randn(bs, 8, chs, seq_len)
sends_db = torch.tensor([0.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0]).view(1, 8, 1)
sends_db = sends_db.repeat(bs, 1, 1)

print(x.shape)

y = stereo_bus(
    x,
    sends_db,
)

print(y.shape)
y /= y.abs().max().clamp(min=1e-8)
torchaudio.save("tests/reverb.wav", y.view(2, -1), sample_rate)
