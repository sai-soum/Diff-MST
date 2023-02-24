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

    return mix, param_dict


def knowledge_engineering_mix(
    tracks: torch.Tensor, mix_console: torch.nn.Module, track_metadata
):
    """Generate a mix using knowledge engineering
    """


    bs, num_tracks, seq_len = tracks.size()
    param_dict = {}
    if mix_console=="BasicMixConsole":
        
        param_dict["input_gain"]

        param_dict = {
            "input_gain": {
                "gain_db":   # bs, num_tracks, 1
            },
            "stereo_panner": {
                "pan": 
            },
        }
    elif mix_console=="AdvancedMixConsole":
       param_dict = {
            "input_gain": {
                "gain_db": 
            },
            "parametric_eq": {
                "low_shelf_gain_db": 
                "low_shelf_cutoff_freq":
                "low_shelf_q_factor": 
                "first_band_gain_db": mix_params[..., 4],
                "first_band_cutoff_freq": mix_params[..., 5],
                "first_band_q_factor": mix_params[..., 6],
                "second_band_gain_db": mix_params[..., 7],
                "second_band_cutoff_freq": mix_params[..., 8],
                "second_band_q_factor": mix_params[..., 9],
                "third_band_gain_db": mix_params[..., 10],
                "third_band_cutoff_freq": mix_params[..., 11],
                "third_band_q_factor": mix_params[..., 12],
                "fourth_band_gain_db": mix_params[..., 13],
                "fourth_band_cutoff_freq": mix_params[..., 14],
                "fourth_band_q_factor": mix_params[..., 15],
                "high_shelf_gain_db": mix_params[..., 16],
                "high_shelf_cutoff_freq": mix_params[..., 17],
                "high_shelf_q_factor": mix_params[..., 18],
            },
            "compressor": {
                "threshold_db": mix_params[..., 19],
                "ratio": mix_params[..., 20],
                "attack_ms": mix_params[..., 21],
                "release_ms": mix_params[..., 22],
                "knee_db": mix_params[..., 23],
                "makeup_gain_db": mix_params[..., 24],
            },
            "stereo_panner": {
                "pan": mix_params[..., 25],
            },
        }
    else:
        raise MixingConsole_Not_Found

    mix = mix_console(tracks, param_dict)

    return mix, param_dict
