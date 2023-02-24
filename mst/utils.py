import torch
import random
import numpy as np


def denorm(p, p_min=0.0, p_max=1.0):
    return (p * (p_max - p_min)) + p_min


def norm(p, p_min=0.0, p_max=1.0):
    return (p - p_min) / (p_max - p_min)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def center_crop(x, length: int):
    start = (x.shape[-1] - length) // 2
    stop = start + length
    return x[..., start:stop]


def causal_crop(x, length: int):
    stop = x.shape[-1] - 1
    start = stop - length
    return x[..., start:stop]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rand(low=0, high=1):
    return (torch.rand(1).numpy()[0] * (high - low)) + low


def randint(low=0, high=1):
    return torch.randint(low, high + 1, (1,)).numpy()[0]


def center_crop(x, length: int):
    if x.shape[-1] != length:
        start = (x.shape[-1] - length) // 2
        stop = start + length
        x = x[..., start:stop]
    return x


def causal_crop(x, length: int):
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x


def find_first_peak(x, threshold_dB=-36, sample_rate=44100):
    """Find the first peak of the input signal.

    Args:
        x (Tensor): 1-d tensor with signal.
        threshold_dB (float, optional): Minimum peak treshold in dB. Default: -36.0
        sample_rate (float, optional): Sample rate of the input signal. Default: 44100

    Returns:
        first_peak_sample (int): Sample index of the first peak.
        first_peak_sec (float): Location of the first peak in seconds.
    """
    signal = 20 * torch.log10(x.view(-1).abs() + 1e-8)
    peaks = torch.where(signal > threshold_dB)[0]
    first_peak_sample = peaks[0]
    first_peak_sec = first_peak_sample / sample_rate

    return first_peak_sample, first_peak_sec


def fade_in_and_fade_out(x, fade_ms=10.0, sample_rate=44100):
    """Apply a linear fade in and fade out to the last dim of a signal.

    Args:
        x (Tensor): Tensor with signal(s).
        fade_ms (float, optional): Length of the fade in milliseconds. Default: 10.0
        sample_rate (int, optional): Sample rate. Default: 44100

    Returns:
        x (Tensor): Faded signal(s).
    """
    fade_samples = int(fade_ms * 1e-3 * sample_rate)
    fade_in = torch.linspace(0, 1.0, fade_samples)
    fade_out = torch.linspace(1.0, 0, fade_samples)
    x[..., :fade_samples] *= fade_in
    x[..., x.shape[-1] - fade_samples :] *= fade_out

    return x


def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if a_set & b_set:
        return True
    else:
        return False
