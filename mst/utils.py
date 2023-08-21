import os
import yaml
import torch
import random
import numpy as np
from importlib import import_module
from mst.modules import MixStyleTransferModel


def batch_stereo_peak_normalize(x: torch.Tensor):
    """Normalize a batch of stereo mixes by their peak value.

    Args:
        x (Tensor): 1-d tensor with shape (bs, 2, seq_len).

    Returns:
        x (Tensor): Normalized signal withs shape (vs, 2, seq_len).
    """
    # first find the peaks in each channel
    gain_lin = x.abs().max(dim=-1, keepdim=True)[0]
    # then find the maximum peak across left and right per batch item
    gain_lin = gain_lin.max(dim=-2, keepdim=True)[0]
    # normalize by the maximum peak
    x_norm = x / gain_lin.clamp(1e-8)  # avoid division by zero
    return x_norm


def load_model(config_path: str, ckpt_path: str, map_location: str = "cpu"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # print(config)
    core_model_configs = config["model"]["init_args"]["model"]

    module_path, class_name = core_model_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(**core_model_configs["init_args"])

    submodule_configs = core_model_configs["init_args"]

    # create track encoder module
    module_path, class_name = submodule_configs["track_encoder"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    track_encoder = getattr(module, class_name)(
        **submodule_configs["track_encoder"]["init_args"]
    )

    # create mix encoder module
    module_path, class_name = submodule_configs["mix_encoder"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    mix_encoder = getattr(module, class_name)(
        **submodule_configs["mix_encoder"]["init_args"]
    )

    # create controller module
    module_path, class_name = submodule_configs["controller"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    controller = getattr(module, class_name)(
        **submodule_configs["controller"]["init_args"]
    )

    # create mix console module
    module_path, class_name = submodule_configs["mix_console"]["class_path"].rsplit(
        ".", 1
    )
    module = import_module(module_path)
    mix_console = getattr(module, class_name)(
        **submodule_configs["mix_console"]["init_args"]
    )

    checkpoint = torch.load(ckpt_path, map_location=map_location)

    # load state dicts
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.track_encoder"):
            state_dict[k.replace("model.track_encoder.", "", 1)] = v
    track_encoder.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.mix_encoder"):
            state_dict[k.replace("model.mix_encoder.", "", 1)] = v
    mix_encoder.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.controller"):
            state_dict[k.replace("model.controller.", "", 1)] = v
    controller.load_state_dict(state_dict)

    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model.mix_console"):
            state_dict[k.replace("model.mix_console.", "", 1)] = v
    mix_console.load_state_dict(state_dict)

    model = MixStyleTransferModel(track_encoder, mix_encoder, controller, mix_console)
    model.eval()

    return model


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
