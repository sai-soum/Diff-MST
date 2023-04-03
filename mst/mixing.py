# Store mixing functions here (e.g. knowledge engineering)
import torch


def naive_random_mix(tracks: torch.Tensor, mix_console: torch.nn.Module):
    """Generate a random mix by sampling parameters uniformly on the parameter ranges.

    Args:
        tracks (torch.Tensor):
        mix_console (torch.nn.Module):

    Returns:
        mix (torch.Tensor)
        param_dict (dict):
    """
    bs, num_tracks, seq_len = tracks.size()

    # generate random parameter tensor
    mix_params = torch.rand(bs, num_tracks, mix_console.num_control_params)
    mix_params = mix_params.type_as(tracks)

    # generate a mix of the tracks
    mix, param_dict = mix_console(tracks, mix_params)

    # peak normalize the mix
    mix /= mix.abs().max().clamp(min=1e-8)

    return mix, param_dict
