# Store mixing functions here (e.g. knowledge engineering)
import json
import torch
import os
import json
import random
import numpy as np
import mst.modules
from mst.modules import BasicMixConsole, AdvancedMixConsole

import mst.dataloaders.medley
from mst.dataloaders.cambridge import CambridgeDataset
from yaml import load, dump, Loader, Dumper
from mst.dataloaders.medley import MedleyDBDataset
import torchaudio
import random


def instrument_metadata(
    instrument_id: list,
    instrument_number_file: dict,
):
    """Convert the metadata info into instrument names"""

    # iid = instrument_id.cpu().tolist()
    bs, num_tracks = instrument_id.size()
    mdata = []
    for i in range(bs):
        iid = instrument_id[i, :]
        # print(iid)
        iid = iid.cpu().tolist()
        metadata = {}
        for j, id in enumerate(iid):
            instrument = [
                instrument
                for instrument, number in instrument_number_file.items()
                if number == id
            ]
            metadata[j] = instrument[0]

        # metadata = dict(zip(range(len(metadata)), metadata))
        # print("metadata:", metadata)
        mdata.append(metadata)
    # print("mdata:", mdata)
    return mdata


def naive_random_mix(tracks: torch.Tensor, mix_console: torch.nn.Module, *args):
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
    mix_tracks, mix, param_dict = mix_console(tracks, mix_params)

    # normalize mix
    gain_lin = 1 / mix.abs().max().clamp(min=1e-8)
    mix *= gain_lin
    mix_tracks *= gain_lin

    return mix_tracks, mix, param_dict


def knowledge_engineering_mix(
    tracks: torch.Tensor,
    mix_console: torch.nn.Module,
    instrument_id: list,
    stereo_id: list,
    instrument_number_file: dict,
    ke_dict: dict,
):
    """Generate a mix using knowledge engineering"""

    KE = ke_dict

    bs, num_tracks, seq_len = tracks.size()
    mdata = instrument_metadata(instrument_id, instrument_number_file)
    # print("mdata:", mdata)

    # BasicMixConsole

    advance_console = False
    """
    Param ranges as defined in mst/modules.py
    min_gain_db: float = -24.0,
    max_gain_db: float = 24.0,
    eq_min_gain_db: float = -24.0,
    eq_max_gain_db: float = 24.0,
    min_pan: float = 0.0,
    max_pan: float = 1.0,
    param_ranges = {
            "input_gain": {"gain_db": (min_gain_db, max_gain_db)},
            "parametric_eq": {
                "low_shelf_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "low_shelf_cutoff_freq": (20, 2000),
                "low_shelf_q_factor": (0.1, 5.0),
                "band0_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band0_cutoff_freq": (80, 2000),
                "band0_q_factor": (0.1, 5.0),
                "band1_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band1_cutoff_freq": (2000, 8000),
                "band1_q_factor": (0.1, 5.0),
                "band2_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band2_cutoff_freq": (8000, 12000),
                "band2_q_factor": (0.1, 5.0),
                "band3_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band3_cutoff_freq": (12000, (sample_rate // 2) - 1000),
                "band3_q_factor": (0.1, 5.0),
                "high_shelf_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "high_shelf_cutoff_freq": (6000, (sample_rate // 2) - 1000),
                "high_shelf_q_factor": (0.1, 5.0),
            },
            "compressor": {
                "threshold_db": (-60.0, 0.0),
                "ratio": (1.0, 10.0),
                "attack_ms": (1.0, 1000.0),
                "release_ms": (1.0, 1000.0),
                "knee_db": (3.0, 24.0),
                "makeup_gain_db": (0.0, 24.0),
            },
            "stereo_panner": {"pan": (min_pan, max_pan)}
    """

    inst_keys = KE.keys()
    # print("inst_keys:", inst_keys)
    mix_params = torch.full((bs, num_tracks, mix_console.num_control_params), -18.0)
    pan_params = torch.full((bs, num_tracks, 1), 0.5)

    if mix_console.num_control_params > 2:
        advance_console = True

        eq_lowshelf_params = torch.full((bs, num_tracks, 3), 0)
        eq_lowshelf_params[:, :, 1] = 100
        eq_lowshelf_params[:, :, 2] = 1.0

        eq_bandpass0_params = torch.full((bs, num_tracks, 3), 0)
        eq_bandpass0_params[:, :, 1] = 500
        eq_bandpass0_params[:, :, 2] = 1.0

        eq_bandpass1_params = torch.full((bs, num_tracks, 3), 0)
        eq_bandpass1_params[:, :, 1] = 3000
        eq_bandpass1_params[:, :, 2] = 1.0

        eq_bandpass2_params = torch.full((bs, num_tracks, 3), 0)
        eq_bandpass2_params[:, :, 1] = 10000
        eq_bandpass2_params[:, :, 2] = 1.0

        eq_bandpass3_params = torch.full((bs, num_tracks, 3), 0)
        eq_bandpass3_params[:, :, 1] = 13000
        eq_bandpass3_params[:, :, 2] = 1.0

        eq_highshelf_params = torch.full((bs, num_tracks, 3), 0)
        eq_highshelf_params[:, :, 1] = 10000
        eq_highshelf_params[:, :, 2] = 1.0

        comp_params = torch.full((bs, num_tracks, 6), -5)
        comp_params[:, :, 1] = 1.0
        comp_params[:, :, 2] = 1.0
        comp_params[:, :, 3] = 1.0
        comp_params[:, :, 4] = 3.0
        comp_params[:, :, 5] = 0.0

        # dist_params = torch.full((bs, num_tracks, 2),(0,0))
        # reverb_params = torch.full((bs, num_tracks, 2),(0,0))

    # uncomment from here --------------------------

    skip = False
    for j in range(bs):
        metadata = mdata[j]
        stereo_info = stereo_id[j, :]

        for i in range(len(metadata)):
            # print(stereo_info[i])
            if stereo_info[i] == 1:
                if i == num_tracks - 1:
                    # print("last track")
                    skip = False
                else:
                    # print("stereo")
                    skip = True
            # print(metadata[i])
            inst_key = [
                key for key in inst_keys if metadata[i] in KE[key]["instruments"]
            ]
            if inst_key == []:
                print("no key found for", metadata[i])
                continue
            # print(inst_key)
            # print(random.choice(KE[inst_key[0]]['gain']))

            mix_params[j, i, 0] = random.uniform(
                KE[inst_key[0]]["gain"][0], KE[inst_key[0]]["gain"][1]
            )
            if (
                not mix_console.param_ranges["input_gain"]["gain_db"][0]
                <= mix_params[j, i, 0]
                <= mix_console.param_ranges["input_gain"]["gain_db"][1]
            ):
                # raise value error
                print(mix_params[j, i, 0])
                print("gain value out of range")

            pan_params[j, i, 0] = random.choice(KE[inst_key[0]]["pan"])
            if (
                not mix_console.param_ranges["stereo_panner"]["pan"][0]
                <= pan_params[j, i, 0]
                <= mix_console.param_ranges["stereo_panner"]["pan"][1]
            ):
                # raise value error
                print(pan_params[j, i, 0])
                print("pan value out of range")

            if advance_console:
                eq_lowshelf_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_lowshelf_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_lowshelf_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["low_shelf_gain_db"][
                        0
                    ]
                    <= eq_lowshelf_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["low_shelf_gain_db"][1]
                ):
                    # raise value error
                    print(eq_lowshelf_params[j, i, 0])
                    print("eq_lowshelf_gain value out of range")

                eq_lowshelf_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_lowshelf_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_lowshelf_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"][
                        "low_shelf_cutoff_freq"
                    ][0]
                    <= eq_lowshelf_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"][
                        "low_shelf_cutoff_freq"
                    ][1]
                ):
                    # raise value error
                    print(eq_lowshelf_params[j, i, 1])
                    print("eq_lowshelf_freq value out of range")

                eq_lowshelf_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_lowshelf_q"][0],
                    KE[inst_key[0]]["eq"]["eq_lowshelf_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["low_shelf_q_factor"][
                        0
                    ]
                    <= eq_lowshelf_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["low_shelf_q_factor"][
                        1
                    ]
                ):
                    # raise value error
                    print(eq_lowshelf_params[j, i, 2])
                    print("eq_lowshelf_q value out of range")

                eq_bandpass0_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band0_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_band0_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band0_gain_db"][0]
                    <= eq_bandpass0_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["band0_gain_db"][1]
                ):
                    # raise value error
                    print(eq_bandpass0_params[j, i, 0])
                    print("eq_band0_gain value out of range")

                eq_bandpass0_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band0_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_band0_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band0_cutoff_freq"][
                        0
                    ]
                    <= eq_bandpass0_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"]["band0_cutoff_freq"][1]
                ):
                    # raise value error
                    print(eq_bandpass0_params[j, i, 1])
                    print("eq_band0_freq value out of range")

                eq_bandpass0_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band0_q"][0],
                    KE[inst_key[0]]["eq"]["eq_band0_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band0_q_factor"][0]
                    <= eq_bandpass0_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["band0_q_factor"][1]
                ):
                    # raise value error
                    print(eq_bandpass0_params[j, i, 2])
                    print("eq_band0_q value out of range")

                eq_bandpass1_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band1_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_band1_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band1_gain_db"][0]
                    <= eq_bandpass1_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["band1_gain_db"][1]
                ):
                    # raise value error
                    print(eq_bandpass1_params[j, i, 0])
                    print("eq_band1_gain value out of range")

                eq_bandpass1_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band1_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_band1_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band1_cutoff_freq"][
                        0
                    ]
                    <= eq_bandpass1_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"]["band1_cutoff_freq"][1]
                ):
                    # raise value error
                    print(eq_bandpass1_params[j, i, 1])
                    print("eq_band1_freq value out of range")

                eq_bandpass1_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band1_q"][0],
                    KE[inst_key[0]]["eq"]["eq_band1_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band1_q_factor"][0]
                    <= eq_bandpass1_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["band1_q_factor"][1]
                ):
                    # raise value error
                    print(eq_bandpass1_params[j, i, 2])
                    print("eq_band1_q value out of range")

                eq_bandpass2_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band2_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_band2_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band2_gain_db"][0]
                    <= eq_bandpass2_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["band2_gain_db"][1]
                ):
                    # raise value error
                    print(eq_bandpass2_params[j, i, 0])
                    print("eq_band2_gain value out of range")

                eq_bandpass2_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band2_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_band2_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band2_cutoff_freq"][
                        0
                    ]
                    <= eq_bandpass2_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"]["band2_cutoff_freq"][1]
                ):
                    # raise value error
                    print(eq_bandpass2_params[j, i, 1])
                    print("eq_band2_freq value out of range")

                eq_bandpass2_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band2_q"][0],
                    KE[inst_key[0]]["eq"]["eq_band2_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band2_q_factor"][0]
                    <= eq_bandpass2_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["band2_q_factor"][1]
                ):
                    # raise value error
                    print(eq_bandpass2_params[j, i, 2])
                    print("eq_band2_q value out of range")

                eq_bandpass3_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band3_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_band3_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band3_gain_db"][0]
                    <= eq_bandpass3_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["band3_gain_db"][1]
                ):
                    # raise value error
                    print(eq_bandpass3_params[j, i, 0])
                    print("eq_band3_gain value out of range")

                eq_bandpass3_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band3_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_band3_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band3_cutoff_freq"][
                        0
                    ]
                    <= eq_bandpass3_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"]["band3_cutoff_freq"][1]
                ):
                    # raise value error
                    print(eq_bandpass3_params[j, i, 1])
                    print("eq_band3_freq value out of range")

                eq_bandpass3_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_band3_q"][0],
                    KE[inst_key[0]]["eq"]["eq_band3_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["band3_q_factor"][0]
                    <= eq_bandpass3_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["band3_q_factor"][1]
                ):
                    # raise value error
                    print(eq_bandpass3_params[j, i, 2])
                    print("eq_band3_q value out of range")

                eq_highshelf_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_highshelf_gain"][0],
                    KE[inst_key[0]]["eq"]["eq_highshelf_gain"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"]["high_shelf_gain_db"][
                        0
                    ]
                    <= eq_highshelf_params[j, i, 0]
                    <= mix_console.param_ranges["parametric_eq"]["high_shelf_gain_db"][
                        1
                    ]
                ):
                    # raise value error
                    print(eq_highshelf_params[j, i, 0])
                    print("eq_highshelf_gain value out of range")

                eq_highshelf_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_highshelf_freq"][0],
                    KE[inst_key[0]]["eq"]["eq_highshelf_freq"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"][
                        "high_shelf_cutoff_freq"
                    ][0]
                    <= eq_highshelf_params[j, i, 1]
                    <= mix_console.param_ranges["parametric_eq"][
                        "high_shelf_cutoff_freq"
                    ][1]
                ):
                    # raise value error
                    print(eq_highshelf_params[j, i, 1])
                    print("eq_highshelf_freq value out of range")

                eq_highshelf_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["eq"]["eq_highshelf_q"][0],
                    KE[inst_key[0]]["eq"]["eq_highshelf_q"][1],
                )
                if (
                    not mix_console.param_ranges["parametric_eq"][
                        "high_shelf_q_factor"
                    ][0]
                    <= eq_highshelf_params[j, i, 2]
                    <= mix_console.param_ranges["parametric_eq"]["high_shelf_q_factor"][
                        1
                    ]
                ):
                    # raise value error
                    print(eq_highshelf_params[j, i, 2])
                    print("eq_highshelf_q value out of range")

                comp_params[j, i, 0] = random.uniform(
                    KE[inst_key[0]]["compressor"]["threshold_db"][0],
                    KE[inst_key[0]]["compressor"]["threshold_db"][1],
                )
                if (
                    not mix_console.param_ranges["compressor"]["threshold_db"][0]
                    <= comp_params[j, i, 0]
                    <= mix_console.param_ranges["compressor"]["threshold_db"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 0])
                    print("comp_threshold value out of range")

                comp_params[j, i, 1] = random.uniform(
                    KE[inst_key[0]]["compressor"]["ratio"][0],
                    KE[inst_key[0]]["compressor"]["ratio"][1],
                )
                if (
                    not mix_console.param_ranges["compressor"]["ratio"][0]
                    <= comp_params[j, i, 1]
                    <= mix_console.param_ranges["compressor"]["ratio"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 1])
                    print("comp_ratio value out of range")

                comp_params[j, i, 2] = random.uniform(
                    KE[inst_key[0]]["compressor"]["attack_ms"][0],
                    KE[inst_key[0]]["compressor"]["attack_ms"][1],
                )
                if (
                    not mix_console.param_ranges["compressor"]["attack_ms"][0]
                    <= comp_params[j, i, 2]
                    <= mix_console.param_ranges["compressor"]["attack_ms"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 2])
                    print("comp_attack value out of range")

                comp_params[j, i, 3] = comp_params[j, i, 2]
                if (
                    not mix_console.param_ranges["compressor"]["release_ms"][0]
                    <= comp_params[j, i, 3]
                    <= mix_console.param_ranges["compressor"]["release_ms"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 3])
                    print("comp_release value out of range")

                comp_params[j, i, 4] = random.uniform(
                    KE[inst_key[0]]["compressor"]["knee_db"][0],
                    KE[inst_key[0]]["compressor"]["knee_db"][1],
                )
                if (
                    not mix_console.param_ranges["compressor"]["knee_db"][0]
                    <= comp_params[j, i, 4]
                    <= mix_console.param_ranges["compressor"]["knee_db"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 4])
                    print("comp_knee value out of range")

                comp_params[j, i, 5] = random.uniform(
                    KE[inst_key[0]]["compressor"]["makeup_gain_db"][0],
                    KE[inst_key[0]]["compressor"]["makeup_gain_db"][1],
                )
                if (
                    not mix_console.param_ranges["compressor"]["makeup_gain_db"][0]
                    <= comp_params[j, i, 5]
                    <= mix_console.param_ranges["compressor"]["makeup_gain_db"][1]
                ):
                    # raise value error
                    print(comp_params[j, i, 5])
                    print("comp_makeup value out of range")

            if skip:
                mix_params[j, i + 1, :] = mix_params[j, i, :]
                pan_params[j, i + 1, :] = 1.0 - pan_params[j, i, :]
                eq_lowshelf_params[j, i + 1, :] = eq_lowshelf_params[j, i, :]
                eq_bandpass0_params[j, i + 1, :] = eq_bandpass0_params[j, i, :]
                eq_bandpass1_params[j, i + 1, :] = eq_bandpass1_params[j, i, :]
                eq_bandpass2_params[j, i + 1, :] = eq_bandpass2_params[j, i, :]
                eq_bandpass3_params[j, i + 1, :] = eq_bandpass3_params[j, i, :]
                eq_highshelf_params[j, i + 1, :] = eq_highshelf_params[j, i, :]
                comp_params[j, i + 1, :] = comp_params[j, i, :]

                i = i + 1
                skip = False

    if mix_console.num_control_params == 2:
        mix_params[:, :, 1] = pan_params[:, :, 0]
        mix_params = mix_params.type_as(tracks)
        param_dict = {
            "input_gain": {
                "gain_db": mix_params[:, :, 0],  # bs, num_tracks, 1
            },
            "stereo_panner": {
                "pan": mix_params[:, :, 1],  # bs, num_tracks, 1
            },
        }

    # applying knowledge engineered compressor and EQ
    elif mix_console.num_control_params != 2:
        # print(eq_lowshelf_params.shape)
        # print(mix_params[:,:,1:4].shape)
        mix_params[:, :, 25] = pan_params[:, :, 0]
        mix_params[:, :, 1:4] = eq_lowshelf_params
        mix_params[:, :, 4:7] = eq_bandpass0_params
        mix_params[:, :, 7:10] = eq_bandpass1_params
        mix_params[:, :, 10:13] = eq_bandpass2_params
        mix_params[:, :, 13:16] = eq_bandpass3_params
        mix_params[:, :, 16:19] = eq_highshelf_params
        mix_params[:, :, 19:25] = comp_params
        mix_params = mix_params.type_as(tracks)
        param_dict = {
            "input_gain": {
                "gain_db": mix_params[..., 0],
            },
            "parametric_eq": {
                "low_shelf_gain_db": mix_params[..., 1],
                "low_shelf_cutoff_freq": mix_params[..., 2],
                "low_shelf_q_factor": mix_params[..., 3],
                "band0_gain_db": mix_params[..., 4],
                "band0_cutoff_freq": mix_params[..., 5],
                "band0_q_factor": mix_params[..., 6],
                "band1_gain_db": mix_params[..., 7],
                "band1_cutoff_freq": mix_params[..., 8],
                "band1_q_factor": mix_params[..., 9],
                "band2_gain_db": mix_params[..., 10],
                "band2_cutoff_freq": mix_params[..., 11],
                "band2_q_factor": mix_params[..., 12],
                "band3_gain_db": mix_params[..., 13],
                "band3_cutoff_freq": mix_params[..., 14],
                "band3_q_factor": mix_params[..., 15],
                "high_shelf_gain_db": mix_params[..., 16],
                "high_shelf_cutoff_freq": mix_params[..., 17],
                "high_shelf_q_factor": mix_params[..., 18],
            },
            # release and attack time must be the same
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

    # check param_dict for out of range parameters
    for effect_name, effect_param_dict in param_dict.items():
        for param_name, param_val in effect_param_dict.items():
            if param_val.min() < mix_console.param_ranges[effect_name][param_name][0]:
                print(f"{param_name} out of range {param_val.min()}")
                print(mix_console.param_ranges[effect_name][param_name][0])
            if param_val.max() > mix_console.param_ranges[effect_name][param_name][1]:
                mix_console.param_ranges[effect_name][param_name][1]
                print(
                    f"{param_name} = out of range.  ({param_val.min()},{param_val.max()})"
                )
                print()

    mix = mix_console.forward_mix_console(tracks, param_dict)
    # peak normalize the mix
    mix /= mix.abs().max().clamp(min=1e-8)

    return mix, param_dict


# if __name__ == "__main__":


#     dataset = CambridgeDataset(root_dirs=["/import/c4dm-multitrack-private/C4DM Multitrack Collection/mixing-secrets"])
#     #dataset = MedleyDBDataset(root_dirs=["/import/c4dm-datasets/MedleyDB_V1/V1"])
#     print(len(dataset))

#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
#     #mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
#     mix_console = mst.modules.AdvancedMixConsole(sample_rate=44100.0)
#     for i, (track, instrument_id, stereo) in enumerate(dataloader):
#         print("\n\n mixing")
#         print("track", track.size())
#         batch_size, num_tracks, seq_len = track.size()


#         # print(instrument_id)
#         # print(stereo)

#         naive_mix, param_dict = naive_random_mix(track, mix_console)
#         print("naive_mix", naive_mix.size())
#         mix, param_dict = knowledge_engineering_mix(track, mix_console, instrument_id, stereo)

#         track = track.view(batch_size,num_tracks,1, seq_len)
#         #print("track", track.size())


#         sum_mix =  torch.sum(track, dim=1)
#         # print("mix", mix.size())
#         #print("sum_mix", sum_mix.size())

#         if not os.path.exists("mix_KE_adv"):
#             os.mkdir("mix_KE_adv")
#         save_dir = "mix_KE_adv/"

#         #export audio
#         for j in range(batch_size):
#             torchaudio.save(os.path.join(save_dir,"mix_"+str(j)+".wav"), mix[j], 44100)
#             torchaudio.save(os.path.join(save_dir,"sum"+str(j)+".wav"), sum_mix[j], 44100)
#             torchaudio.save(os.path.join(save_dir,"naive"+str(j)+".wav"), naive_mix[j], 44100)
#         print("mix", mix.size())
#         if i==0:
#             break
