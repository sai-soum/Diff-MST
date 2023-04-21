import torch
import torchaudio
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from dasp_pytorch.functional import parametric_eq


bs = 2
chs = 2
seq_len = 131072
sample_rate = 44100

x = torch.zeros(bs, chs, seq_len)
x = x / x.abs().max().clamp(min=1e-8)
x *= 10 ** (-24 / 20)

low_shelf_gain_db = torch.tensor([0.0, 0.0])
low_shelf_cutoff_freq = torch.tensor([20.0, 20.0])
low_shelf_q_factor = torch.tensor([0.707, 0.707])
band0_gain_db = torch.tensor([0.0, 12.0])
band0_cutoff_freq = torch.tensor([100.0, 100.0])
band0_q_factor = torch.tensor([0.707, 0.707])
band1_gain_db = torch.tensor([12.0, 0.0])
band1_cutoff_freq = torch.tensor([1000.0, 1000.0])
band1_q_factor = torch.tensor([6.0, 0.707])
band2_gain_db = torch.tensor([-6.0, 0.0])
band2_cutoff_freq = torch.tensor([10000.0, 10000.0])
band2_q_factor = torch.tensor([3.0, 0.707])
band3_gain_db = torch.tensor([0.0, 0.0])
band3_cutoff_freq = torch.tensor([12000.0, 12000.0])
band3_q_factor = torch.tensor([0.707, 0.707])
high_shelf_gain_db = torch.tensor([0.0, 0.0])
high_shelf_cutoff_freq = torch.tensor([12000.0, 12000.0])
high_shelf_q_factor = torch.tensor([0.707, 0.707])

# reshape and repeat to match batch size
low_shelf_gain_db = low_shelf_gain_db.view(1, chs).repeat(bs, 1)
low_shelf_cutoff_freq = low_shelf_cutoff_freq.view(1, chs).repeat(bs, 1)
low_shelf_q_factor = low_shelf_q_factor.view(1, chs).repeat(bs, 1)
band0_gain_db = band0_gain_db.view(1, chs).repeat(bs, 1)
band0_cutoff_freq = band0_cutoff_freq.view(1, chs).repeat(bs, 1)
band0_q_factor = band0_q_factor.view(1, chs).repeat(bs, 1)
band1_gain_db = band1_gain_db.view(1, chs).repeat(bs, 1)
band1_cutoff_freq = band1_cutoff_freq.view(1, chs).repeat(bs, 1)
band1_q_factor = band1_q_factor.view(1, chs).repeat(bs, 1)
band2_gain_db = band2_gain_db.view(1, chs).repeat(bs, 1)
band2_cutoff_freq = band2_cutoff_freq.view(1, chs).repeat(bs, 1)
band2_q_factor = band2_q_factor.view(1, chs).repeat(bs, 1)
band3_gain_db = band3_gain_db.view(1, chs).repeat(bs, 1)
band3_cutoff_freq = band3_cutoff_freq.view(1, chs).repeat(bs, 1)
band3_q_factor = band3_q_factor.view(1, chs).repeat(bs, 1)
high_shelf_gain_db = high_shelf_gain_db.view(1, chs).repeat(bs, 1)
high_shelf_cutoff_freq = high_shelf_cutoff_freq.view(1, chs).repeat(bs, 1)
high_shelf_q_factor = high_shelf_q_factor.view(1, chs).repeat(bs, 1)

y = parametric_eq(
    x,
    sample_rate,
    low_shelf_gain_db,
    low_shelf_cutoff_freq,
    low_shelf_q_factor,
    band0_gain_db,
    band0_cutoff_freq,
    band0_q_factor,
    band1_gain_db,
    band1_cutoff_freq,
    band1_q_factor,
    band2_gain_db,
    band2_cutoff_freq,
    band2_q_factor,
    band3_gain_db,
    band3_cutoff_freq,
    band3_q_factor,
    high_shelf_gain_db,
    high_shelf_cutoff_freq,
    high_shelf_q_factor,
)

print(y)
print(y.shape)

fig, axs = plt.subplots(chs, 1, figsize=(10, 6))
for ch in range(chs):
    h_in = 20 * torch.log10(torch.fft.rfft(x[0, ch, :], dim=-1).abs() + 1e-8)
    h_out = 20 * torch.log10(torch.fft.rfft(y[0, ch, :], dim=-1).abs() + 1e-8)

    h_in_sm = savgol_filter(h_in.squeeze().numpy(), 255, 3)
    h_out_sm = savgol_filter(h_out.squeeze().numpy(), 255, 3)

    axs[ch].plot(h_out_sm - h_in_sm, label="input")
    axs[ch].set_xscale("log")
    axs[ch].legend()

plt.savefig("test-peq.png", dpi=300)

# torchaudio.save("test-peq-in.wav", x.view(1, -1), sample_rate)
# torchaudio.save("test-peq-out.wav", y.view(1, -1), sample_rate)
