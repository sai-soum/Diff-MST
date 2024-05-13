import os
import yaml
import torch
import random
import numpy as np
import pyloudnorm as pyln

from tqdm import tqdm
from typing import Optional
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


def run_diffmst(
    tracks: torch.Tensor,
    ref: torch.Tensor,
    model: torch.nn.Module,
    mix_console: torch.nn.Module,
    track_start_idx: int = 0,
    ref_start_idx: int = 0,
):
    """Run the differentiable mix style transfer model.

    Args:
        tracks (Tensor): Set of input tracks with shape (bs, num_tracks, 1, seq_len).
        ref (Tensor): Reference mix with shape (bs, 2, seq_len).
        model (torch.nn.Module): MixStyleTransferModel instance.
        mix_console (torch.nn.Module): MixConsole instance.
        track_start_idx (int, optional): Start index of the track to use. Default: 0.
        ref_start_idx (int, optional): Start index of the reference mix to use. Default: 0.

    Returns:
        pred_mix (Tensor): Predicted mix with shape (bs, 2, seq_len).
        pred_track_param_dict (dict): Dictionary with predicted track parameters.
        pred_fx_bus_param_dict (dict): Dictionary with predicted fx bus parameters.
        pred_master_bus_param_dict (dict): Dictionary with predicted master bus parameters.
    """
    # ------ defaults ------
    use_track_input_fader = True
    use_track_panner = True
    use_track_eq = True
    use_track_compressor = True
    use_fx_bus = False
    use_master_bus = True
    use_output_fader = True

    analysis_len = 262144
    meter = pyln.Meter(44100)

    # crop the input tracks and reference mix to the analysis length
    if tracks.shape[-1] >= analysis_len:
        analysis_tracks = tracks[
            ..., track_start_idx : track_start_idx + analysis_len
        ].clone()
    else:
        analysis_tracks = tracks.clone()

    if ref.shape[-1] >= analysis_len:
        analysis_ref = ref[..., ref_start_idx : ref_start_idx + analysis_len]
    else:
        analysis_ref = ref.clone()

    # loudness normalize the tracks to -48 LUFS
    norm_tracks = []
    norm_analysis_tracks = []
    track_padding = []
    for track_idx in range(analysis_tracks.shape[1]):
        analysis_track = analysis_tracks[:, track_idx : track_idx + 1, :]
        track = tracks[:, track_idx : track_idx + 1, :]
        lufs_db = meter.integrated_loudness(
            analysis_track.squeeze(0).permute(1, 0).numpy()
        )
        if lufs_db < -80.0:
            print(f"Skipping track {track_idx} due to low loudness {lufs_db}.")
            continue

        lufs_delta_db = -48 - lufs_db
        analysis_track *= 10 ** (lufs_delta_db / 20)
        track *= 10 ** (lufs_delta_db / 20)

        norm_analysis_tracks.append(analysis_track)
        norm_tracks.append(track)
        track_padding.append(False)

    norm_analysis_tracks = torch.cat(norm_analysis_tracks, dim=1)
    norm_tracks = torch.cat(norm_tracks, dim=1)
    print(norm_analysis_tracks.shape, norm_tracks.shape)

    # take only first 16 tracks
    # norm_tracks = norm_tracks[:, :16, :]
    # norm_analysis_tracks = norm_analysis_tracks[:, :16, :]
    print(norm_analysis_tracks.shape, norm_tracks.shape)

    # make tensor contiguous
    norm_analysis_tracks = norm_analysis_tracks.contiguous()
    norm_tracks = norm_tracks.contiguous()

    #  ---- run model to estimate mix parmaeters using analysis audio ----
    pred_track_params, pred_fx_bus_params, pred_master_bus_params = model(
        norm_analysis_tracks, analysis_ref
    )

    # ------- generate a mix using the predicted mix console parameters -------
    # apply with sliding window of 262144 samples with overlap
    pred_mix = torch.zeros(1, 2, norm_tracks.shape[-1])

    for i in tqdm(range(0, norm_tracks.shape[-1], analysis_len // 2)):
        norm_tracks_window = norm_tracks[..., i : i + analysis_len]
        (
            pred_mixed_tracks,
            pred_mix_window,
            pred_track_param_dict,
            pred_fx_bus_param_dict,
            pred_master_bus_param_dict,
        ) = mix_console(
            norm_tracks_window,
            pred_track_params,
            pred_fx_bus_params,
            pred_master_bus_params,
            use_track_input_fader=use_track_input_fader,
            use_track_panner=use_track_panner,
            use_track_eq=use_track_eq,
            use_track_compressor=use_track_compressor,
            use_fx_bus=use_fx_bus,
            use_master_bus=use_master_bus,
            use_output_fader=use_output_fader,
        )
        if pred_mix_window.shape[-1] < analysis_len:
            pred_mix_window = torch.nn.functional.pad(
                pred_mix_window, (0, analysis_len - pred_mix_window.shape[-1])
            )

        window = torch.hann_window(pred_mix_window.shape[-1])
        # apply hann window
        if i == 0:
            # set the first half of the window to 1
            window[: window.shape[-1] // 2] = 1.0

        pred_mix_window *= window

        # check length of the mix window
        output_len = pred_mix[..., i : i + analysis_len].shape[-1]

        # overlap add
        pred_mix[..., i : i + analysis_len] += pred_mix_window[..., :output_len]

    # crop the mix to the original length
    pred_mix = pred_mix[..., : norm_tracks.shape[-1]]

    return (
        pred_mix,
        pred_track_param_dict,
        pred_fx_bus_param_dict,
        pred_master_bus_param_dict,
    )


def load_diffmst(config_path: str, ckpt_path: str, map_location: str = "cpu"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

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
    module_path, class_name = config["model"]["init_args"]["mix_console"][
        "class_path"
    ].rsplit(".", 1)
    module = import_module(module_path)
    mix_console = getattr(module, class_name)(
        **config["model"]["init_args"]["mix_console"]["init_args"]
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

    model = MixStyleTransferModel(
        track_encoder,
        mix_encoder,
        controller,
    )
    model.eval()

    return model, mix_console


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
