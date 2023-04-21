import torch
import torchaudio
import matplotlib.pyplot as plt

from dasp_pytorch.functional import compressor


bs = 2
chs = 2
seq_len = 262144
sample_rate = 44100

# x = torch.randn(bs, chs, seq_len)
# x = x / x.abs().max().clamp(min=1e-8)
# x *= 10 ** (-24 / 20)


x = torch.zeros(bs, chs, seq_len)

x[..., 0, 4096:131072] = 1.0
x[..., 1, 16384:65536] = 1.0


threshold_db = torch.tensor([-12.0, -6.0])
ratio = torch.tensor([4.0, 4.0])
attack_ms = torch.tensor([100.0, 1000.0])
release_ms = torch.tensor([0.0, 0.0])  # dummy parameter
knee_db = torch.tensor([6.0, 6.0])
makeup_gain_db = torch.tensor([0.0, 0.0])

threshold_db = threshold_db.view(1, chs).repeat(bs, 1)
ratio = ratio.view(1, chs).repeat(bs, 1)
attack_ms = attack_ms.view(1, chs).repeat(bs, 1)
release_ms = release_ms.view(1, chs).repeat(bs, 1)
knee_db = knee_db.view(1, chs).repeat(bs, 1)
makeup_gain_db = makeup_gain_db.view(1, chs).repeat(bs, 1)

print(threshold_db.shape)
y = compressor(
    x,
    sample_rate,
    threshold_db,
    ratio,
    attack_ms,
    release_ms,
    knee_db,
    makeup_gain_db,
)

print(y.shape)

fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(x[0, 0, :].numpy(), label="input")
axs[0].plot(y[0, 0, :].numpy(), label="output")
axs[0].legend()

axs[1].plot(x[0, 1, :].numpy(), label="input")
axs[1].plot(y[0, 1, :].numpy(), label="output")
axs[1].legend()
plt.grid(c="lightgray")
plt.savefig("compressor.png", dpi=300)
