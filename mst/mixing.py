# Store mixing functions here (e.g. knowledge engineering)
import torch
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


def naive_random_mix(
    tracks: torch.Tensor,
    mix_console: torch.nn.Module,
    use_track_input_fader: bool = True,
    use_track_eq: bool = True,
    use_track_compressor: bool = True,
    use_track_panner: bool = True,
    use_fx_bus: bool = True,
    use_master_bus: bool = True,
    use_ouput_fader: bool = True,
    **kwargs,
):
    """Generate a random mix by sampling parameters uniformly on the parameter ranges.

    Args:
        tracks (torch.Tensor):
        mix_console (torch.nn.Module):
        global_step (int): Global step of the training loop. Used to determine the effects that are active.

    Returns:
        mix (torch.Tensor)
        param_dict (dict):
    """
    bs, num_tracks, seq_len = tracks.size()

    # generate random parameter tensors
    mix_params = torch.rand(bs, num_tracks, mix_console.num_track_control_params)
    mix_params = mix_params.type_as(tracks)
    # print("mix_params in the dataset making:", mix_params)

    fx_bus_params = torch.rand(bs, mix_console.num_fx_bus_control_params)
    fx_bus_params = fx_bus_params.type_as(tracks)

    master_bus_params = torch.rand(bs, mix_console.num_master_bus_control_params)
    master_bus_params = master_bus_params.type_as(tracks)

    # ------------ generate a mix of the tracks ------------
    with torch.no_grad():
        (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        ) = mix_console(
            tracks,
            mix_params,
            fx_bus_params,
            master_bus_params,
            use_track_input_fader=use_track_input_fader,
            use_track_eq=use_track_eq,
            use_track_compressor=use_track_compressor,
            use_track_panner=use_track_panner,
            use_master_bus=use_master_bus,
            use_fx_bus=use_fx_bus,
            use_output_fader=use_ouput_fader,
        )


    return mixed_tracks, mix, track_param_dict, fx_bus_param_dict, master_bus_param_dict, mix_params, fx_bus_params, master_bus_params



def knowledge_engineering_mix(
    tracks: torch.Tensor,
    mix_console: torch.nn.Module,
    instrument_id: list,
    stereo_id: list,
    instrument_number_file: dict,
    ke_dict: dict,
    use_track_gain: bool = True,
    use_track_eq: bool = True,
    use_track_compressor: bool = True,
    use_track_panner: bool = True,
    use_fx_bus: bool = True,
    use_master_bus: bool = True,
    sample_rate: int = 44100,
    warmup: int = 0,
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
    # Set defaullt calues for gain and pan (centre panning and min gain and intialize other params to 0)

    mix_params = torch.full(
        (bs, num_tracks, mix_console.num_track_control_params), -18.0
    )

    pan_params = torch.full((bs, num_tracks, 1), 0.5)

    # In case the number of parameters is more than 2, we are working with advanced console, so set each of the value to the default min from the range
    if mix_console.num_track_control_params > 2:
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
        comp_params[:, :, 2] = 10.0
        comp_params[:, :, 3] = 10.0
        comp_params[:, :, 4] = 3.0
        comp_params[:, :, 5] = 0.0

        fx_send_params = torch.full((bs, num_tracks, 1), 0.0)

        # fx_band_gain = torch.full((bs,12))
        # fx_band_decay = torch.full((bs,12))
        # fx_mix = torch.full((bs,1))
        # fx_send_db = torch.full((bs,1))
        fx_params = torch.full((bs, mix_console.num_fx_bus_control_params), 0.0)
        # 12-band gain parameters
        fx_params[:, 0:12] = 0.0
        # 12-band decay parameters
        fx_params[:, 12:24] = 0.0
        # mix parameter
        fx_params[:, 24] = 0.0

        master_params = torch.full((bs, mix_console.num_master_bus_control_params), 0.0)
        # eq params
        # low shelf
        master_params[:, 1] = 100
        master_params[:, 2] = 1.0

        # bandpass0
        master_params[:, 3] = 0.0
        master_params[:, 4] = 500
        master_params[:, 5] = 1.0

        # bandpass1

        master_params[:, 6] = 0.0
        master_params[:, 7] = 3000
        master_params[:, 8] = 1.0

        # bandpass2
        master_params[:, 9] = 0.0
        master_params[:, 10] = 10000
        master_params[:, 11] = 1.0

        # bandpass3
        master_params[:, 12] = 0.0
        master_params[:, 13] = sample_rate / 2
        master_params[:, 14] = 1.0

        # high shelf
        master_params[:, 15] = 0.0
        master_params[:, 16] = sample_rate / 2
        master_params[:, 17] = 1.0

        # compressor
        master_params[:, 18] = -5
        master_params[:, 19] = 1.0
        master_params[:, 20] = 10.0
        master_params[:, 21] = 10.0
        master_params[:, 22] = 3.0
        master_params[:, 23] = 0.0

        # master gain
        master_params[:, 24] = -10.0

    # skip is set to true whenever a track is stereo as it is loaded onto two tracks from the dataset one each for left and right channel
    skip = False
    # We work with each song in the batch seperately. It is sort of not possible to do it simultaneously for all the songs in the batch
    for j in range(bs):
        # We need instrument info to set the values for parameters
        metadata = mdata[j]
        # This is to check if the track is stereo or not. If the stereo_id is 1 that means the next track is basically the other channel of the same track, hence needs to have same settings.
        stereo_info = stereo_id[j, :]

        for i in range(len(metadata)):
            # print(stereo_info[i])
            # if the last track happens to be stereo, then skip value need not be true.
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
            # key values as returned from dataloader are numbers, so we need to convert them to strings(instrument names and then use the info to make mixing decisions)
            if inst_key == []:
                print("no key found for", metadata[i])
                continue
            # print(inst_key)
            # print(random.choice(KE[inst_key[0]]['gain']))
            # since the yaml file stores ranges, we use random function to assign a value in that range just to create some diversity

            mix_params[j, i, 0] = random.uniform(
                KE[inst_key[0]]["gain"][0], KE[inst_key[0]]["gain"][1]
            )
            if (
                not mix_console.param_ranges["fader"]["gain_db"][0]
                <= mix_params[j, i, 0]
                <= mix_console.param_ranges["fader"]["gain_db"][1]
            ):
                # raise value error
                print(mix_params[j, i, 0])
                print("gain value out of range")
            # Pan values are stored as individual numbers in the yaml file, so we use random.choice to select one of them

            pan_params[j, i, 0] = random.choice(KE[inst_key[0]]["pan"])
            if (
                not mix_console.param_ranges["stereo_panner"]["pan"][0]
                <= pan_params[j, i, 0]
                <= mix_console.param_ranges["stereo_panner"]["pan"][1]
            ):
                # raise value error
                print(pan_params[j, i, 0])
                print("pan value out of range")
            # If the number of parameters is more than 2, we are working with advanced console, then we apply other effects
            if advance_console:
                if use_track_eq == True:
                    eq_lowshelf_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_lowshelf_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_lowshelf_gain"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"][
                            "low_shelf_gain_db"
                        ][0]
                        <= eq_lowshelf_params[j, i, 0]
                        <= mix_console.param_ranges["parametric_eq"][
                            "low_shelf_gain_db"
                        ][1]
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
                        not mix_console.param_ranges["parametric_eq"][
                            "low_shelf_q_factor"
                        ][0]
                        <= eq_lowshelf_params[j, i, 2]
                        <= mix_console.param_ranges["parametric_eq"][
                            "low_shelf_q_factor"
                        ][1]
                    ):
                        # raise value error
                        print(eq_lowshelf_params[j, i, 2])
                        print("eq_lowshelf_q value out of range")

                    eq_bandpass0_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band0_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_band0_gain"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band0_gain_db"][
                            0
                        ]
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
                        not mix_console.param_ranges["parametric_eq"][
                            "band0_cutoff_freq"
                        ][0]
                        <= eq_bandpass0_params[j, i, 1]
                        <= mix_console.param_ranges["parametric_eq"][
                            "band0_cutoff_freq"
                        ][1]
                    ):
                        # raise value error
                        print(eq_bandpass0_params[j, i, 1])
                        print("eq_band0_freq value out of range")

                    eq_bandpass0_params[j, i, 2] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band0_q"][0],
                        KE[inst_key[0]]["eq"]["eq_band0_q"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band0_q_factor"][
                            0
                        ]
                        <= eq_bandpass0_params[j, i, 2]
                        <= mix_console.param_ranges["parametric_eq"]["band0_q_factor"][
                            1
                        ]
                    ):
                        # raise value error
                        print(eq_bandpass0_params[j, i, 2])
                        print("eq_band0_q value out of range")

                    eq_bandpass1_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band1_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_band1_gain"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band1_gain_db"][
                            0
                        ]
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
                        not mix_console.param_ranges["parametric_eq"][
                            "band1_cutoff_freq"
                        ][0]
                        <= eq_bandpass1_params[j, i, 1]
                        <= mix_console.param_ranges["parametric_eq"][
                            "band1_cutoff_freq"
                        ][1]
                    ):
                        # raise value error
                        print(eq_bandpass1_params[j, i, 1])
                        print("eq_band1_freq value out of range")

                    eq_bandpass1_params[j, i, 2] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band1_q"][0],
                        KE[inst_key[0]]["eq"]["eq_band1_q"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band1_q_factor"][
                            0
                        ]
                        <= eq_bandpass1_params[j, i, 2]
                        <= mix_console.param_ranges["parametric_eq"]["band1_q_factor"][
                            1
                        ]
                    ):
                        # raise value error
                        print(eq_bandpass1_params[j, i, 2])
                        print("eq_band1_q value out of range")

                    eq_bandpass2_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band2_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_band2_gain"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band2_gain_db"][
                            0
                        ]
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
                        not mix_console.param_ranges["parametric_eq"][
                            "band2_cutoff_freq"
                        ][0]
                        <= eq_bandpass2_params[j, i, 1]
                        <= mix_console.param_ranges["parametric_eq"][
                            "band2_cutoff_freq"
                        ][1]
                    ):
                        # raise value error
                        print(eq_bandpass2_params[j, i, 1])
                        print("eq_band2_freq value out of range")

                    eq_bandpass2_params[j, i, 2] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band2_q"][0],
                        KE[inst_key[0]]["eq"]["eq_band2_q"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band2_q_factor"][
                            0
                        ]
                        <= eq_bandpass2_params[j, i, 2]
                        <= mix_console.param_ranges["parametric_eq"]["band2_q_factor"][
                            1
                        ]
                    ):
                        # raise value error
                        print(eq_bandpass2_params[j, i, 2])
                        print("eq_band2_q value out of range")

                    eq_bandpass3_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band3_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_band3_gain"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band3_gain_db"][
                            0
                        ]
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
                    if eq_bandpass3_params[j, i, 1] > sample_rate / 2:
                        eq_bandpass3_params = sample_rate / 2
                    if (
                        not mix_console.param_ranges["parametric_eq"][
                            "band3_cutoff_freq"
                        ][0]
                        <= eq_bandpass3_params[j, i, 1]
                        <= mix_console.param_ranges["parametric_eq"][
                            "band3_cutoff_freq"
                        ][1]
                    ):
                        # raise value error
                        print(eq_bandpass3_params[j, i, 1])
                        print("eq_band3_freq value out of range")

                    eq_bandpass3_params[j, i, 2] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_band3_q"][0],
                        KE[inst_key[0]]["eq"]["eq_band3_q"][1],
                    )
                    if (
                        not mix_console.param_ranges["parametric_eq"]["band3_q_factor"][
                            0
                        ]
                        <= eq_bandpass3_params[j, i, 2]
                        <= mix_console.param_ranges["parametric_eq"]["band3_q_factor"][
                            1
                        ]
                    ):
                        # raise value error
                        print(eq_bandpass3_params[j, i, 2])
                        print("eq_band3_q value out of range")

                    eq_highshelf_params[j, i, 0] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_highshelf_gain"][0],
                        KE[inst_key[0]]["eq"]["eq_highshelf_gain"][1],
                    )

                    if (
                        not mix_console.param_ranges["parametric_eq"][
                            "high_shelf_gain_db"
                        ][0]
                        <= eq_highshelf_params[j, i, 0]
                        <= mix_console.param_ranges["parametric_eq"][
                            "high_shelf_gain_db"
                        ][1]
                    ):
                        # raise value error
                        print(eq_highshelf_params[j, i, 0])
                        print("eq_highshelf_gain value out of range")

                    eq_highshelf_params[j, i, 1] = random.uniform(
                        KE[inst_key[0]]["eq"]["eq_highshelf_freq"][0],
                        KE[inst_key[0]]["eq"]["eq_highshelf_freq"][1],
                    )
                    if eq_highshelf_params[j, i, 1] > sample_rate / 2:
                        eq_highshelf_params[j, i, 1] = sample_rate / 2
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
                        <= mix_console.param_ranges["parametric_eq"][
                            "high_shelf_q_factor"
                        ][1]
                    ):
                        # raise value error
                        print(eq_highshelf_params[j, i, 2])
                        print("eq_highshelf_q value out of range")
                if use_track_compressor == True:
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
                    fx_send_params[j, i, 0] = random.uniform(
                        KE["fx_bus"]["send_db"][0], KE["fx_bus"]["send_db"][1]
                    )
                # if skip was true, implies the next track is a different channel of the same track, hence set the same values
            if skip:
                mix_params[j, i + 1, :] = mix_params[j, i, :]

                pan_params[j, i + 1, :] = 1.0 - pan_params[j, i, :]
                if use_track_eq == True:
                    eq_lowshelf_params[j, i + 1, :] = eq_lowshelf_params[j, i, :]
                    eq_bandpass0_params[j, i + 1, :] = eq_bandpass0_params[j, i, :]
                    eq_bandpass1_params[j, i + 1, :] = eq_bandpass1_params[j, i, :]
                    eq_bandpass2_params[j, i + 1, :] = eq_bandpass2_params[j, i, :]
                    eq_bandpass3_params[j, i + 1, :] = eq_bandpass3_params[j, i, :]
                    eq_highshelf_params[j, i + 1, :] = eq_highshelf_params[j, i, :]
                if use_track_compressor == True:
                    comp_params[j, i + 1, :] = comp_params[j, i, :]
                if use_fx_bus == True:
                    fx_send_params[j, i + 1, :] = fx_send_params[j, i, :]

                i = i + 1
                skip = False
        # every song has only one fx bus; so we only iterate over batch_size for this
        if use_fx_bus:
            fx_params[j, 0] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_0"][0],
                KE["fx_bus"]["reverb_gain"]["band_0"][1],
            )
            fx_params[j, 1] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_1"][0],
                KE["fx_bus"]["reverb_gain"]["band_1"][1],
            )
            fx_params[j, 2] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_2"][0],
                KE["fx_bus"]["reverb_gain"]["band_2"][1],
            )
            fx_params[j, 3] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_3"][0],
                KE["fx_bus"]["reverb_gain"]["band_3"][1],
            )
            fx_params[j, 4] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_4"][0],
                KE["fx_bus"]["reverb_gain"]["band_4"][1],
            )
            fx_params[j, 5] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_5"][0],
                KE["fx_bus"]["reverb_gain"]["band_5"][1],
            )
            fx_params[j, 6] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_6"][0],
                KE["fx_bus"]["reverb_gain"]["band_6"][1],
            )
            fx_params[j, 7] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_7"][0],
                KE["fx_bus"]["reverb_gain"]["band_7"][1],
            )
            fx_params[j, 8] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_8"][0],
                KE["fx_bus"]["reverb_gain"]["band_8"][1],
            )
            fx_params[j, 9] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_9"][0],
                KE["fx_bus"]["reverb_gain"]["band_9"][1],
            )
            fx_params[j, 10] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_10"][0],
                KE["fx_bus"]["reverb_gain"]["band_10"][1],
            )
            fx_params[j, 11] = random.uniform(
                KE["fx_bus"]["reverb_gain"]["band_11"][0],
                KE["fx_bus"]["reverb_gain"]["band_11"][1],
            )

            fx_params[j, 12] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_0"][0],
                KE["fx_bus"]["reverb_decay"]["band_0"][1],
            )
            fx_params[j, 12] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_1"][0],
                KE["fx_bus"]["reverb_decay"]["band_1"][1],
            )
            fx_params[j, 14] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_2"][0],
                KE["fx_bus"]["reverb_decay"]["band_2"][1],
            )
            fx_params[j, 15] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_3"][0],
                KE["fx_bus"]["reverb_decay"]["band_3"][1],
            )
            fx_params[j, 16] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_4"][0],
                KE["fx_bus"]["reverb_decay"]["band_4"][1],
            )
            fx_params[j, 17] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_5"][0],
                KE["fx_bus"]["reverb_decay"]["band_5"][1],
            )
            fx_params[j, 18] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_6"][0],
                KE["fx_bus"]["reverb_decay"]["band_6"][1],
            )
            fx_params[j, 19] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_7"][0],
                KE["fx_bus"]["reverb_decay"]["band_7"][1],
            )
            fx_params[j, 20] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_8"][0],
                KE["fx_bus"]["reverb_decay"]["band_8"][1],
            )
            fx_params[j, 21] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_9"][0],
                KE["fx_bus"]["reverb_decay"]["band_9"][1],
            )
            fx_params[j, 22] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_10"][0],
                KE["fx_bus"]["reverb_decay"]["band_10"][1],
            )
            fx_params[j, 23] = random.uniform(
                KE["fx_bus"]["reverb_decay"]["band_11"][0],
                KE["fx_bus"]["reverb_decay"]["band_11"][1],
            )

            fx_params[j, 24] = random.uniform(
                KE["fx_bus"]["mix"][0], KE["fx_bus"]["mix"][1]
            )

        if use_master_bus:
            master_params[j, 0] = random.uniform(
                KE["master_bus"]["eq"]["eq_lowshelf_gain"][0],
                KE["master_bus"]["eq"]["eq_lowshelf_gain"][1],
            )
            master_params[j, 1] = random.uniform(
                KE["master_bus"]["eq"]["eq_lowshelf_freq"][0],
                KE["master_bus"]["eq"]["eq_lowshelf_freq"][1],
            )
            master_params[j, 2] = random.uniform(
                KE["master_bus"]["eq"]["eq_lowshelf_q"][0],
                KE["master_bus"]["eq"]["eq_lowshelf_q"][1],
            )

            master_params[j, 3] = random.uniform(
                KE["master_bus"]["eq"]["eq_band0_gain"][0],
                KE["master_bus"]["eq"]["eq_band0_gain"][1],
            )
            master_params[j, 4] = random.uniform(
                KE["master_bus"]["eq"]["eq_band0_freq"][0],
                KE["master_bus"]["eq"]["eq_band0_freq"][1],
            )
            master_params[j, 5] = random.uniform(
                KE["master_bus"]["eq"]["eq_band0_q"][0],
                KE["master_bus"]["eq"]["eq_band0_q"][1],
            )

            master_params[j, 6] = random.uniform(
                KE["master_bus"]["eq"]["eq_band1_gain"][0],
                KE["master_bus"]["eq"]["eq_band1_gain"][1],
            )
            master_params[j, 7] = random.uniform(
                KE["master_bus"]["eq"]["eq_band1_freq"][0],
                KE["master_bus"]["eq"]["eq_band1_freq"][1],
            )
            master_params[j, 8] = random.uniform(
                KE["master_bus"]["eq"]["eq_band1_q"][0],
                KE["master_bus"]["eq"]["eq_band1_q"][1],
            )

            master_params[j, 9] = random.uniform(
                KE["master_bus"]["eq"]["eq_band2_gain"][0],
                KE["master_bus"]["eq"]["eq_band2_gain"][1],
            )
            master_params[j, 10] = random.uniform(
                KE["master_bus"]["eq"]["eq_band2_freq"][0],
                KE["master_bus"]["eq"]["eq_band2_freq"][1],
            )
            master_params[j, 11] = random.uniform(
                KE["master_bus"]["eq"]["eq_band2_q"][0],
                KE["master_bus"]["eq"]["eq_band2_q"][1],
            )

            master_params[j, 12] = random.uniform(
                KE["master_bus"]["eq"]["eq_band3_gain"][0],
                KE["master_bus"]["eq"]["eq_band3_gain"][1],
            )
            master_params[j, 13] = random.uniform(
                KE["master_bus"]["eq"]["eq_band3_freq"][0],
                KE["master_bus"]["eq"]["eq_band3_freq"][1],
            )
            master_params[j, 14] = random.uniform(
                KE["master_bus"]["eq"]["eq_band3_q"][0],
                KE["master_bus"]["eq"]["eq_band3_q"][1],
            )

            master_params[j, 15] = random.uniform(
                KE["master_bus"]["eq"]["eq_highshelf_gain"][0],
                KE["master_bus"]["eq"]["eq_highshelf_gain"][1],
            )
            master_params[j, 16] = random.uniform(
                KE["master_bus"]["eq"]["eq_highshelf_freq"][0],
                KE["master_bus"]["eq"]["eq_highshelf_freq"][1],
            )
            master_params[j, 17] = random.uniform(
                KE["master_bus"]["eq"]["eq_highshelf_q"][0],
                KE["master_bus"]["eq"]["eq_highshelf_q"][1],
            )

            master_params[j, 18] = random.uniform(
                KE["master_bus"]["compressor"]["threshold_db"][0],
                KE["master_bus"]["compressor"]["threshold_db"][1],
            )
            master_params[j, 19] = random.uniform(
                KE["master_bus"]["compressor"]["ratio"][0],
                KE["master_bus"]["compressor"]["ratio"][1],
            )
            master_params[j, 20] = random.uniform(
                KE["master_bus"]["compressor"]["attack_ms"][0],
                KE["master_bus"]["compressor"]["attack_ms"][1],
            )
            master_params[j, 21] = random.uniform(
                KE["master_bus"]["compressor"]["release_ms"][0],
                KE["master_bus"]["compressor"]["release_ms"][1],
            )
            master_params[j, 22] = random.uniform(
                KE["master_bus"]["compressor"]["knee_db"][0],
                KE["master_bus"]["compressor"]["knee_db"][1],
            )
            master_params[j, 23] = random.uniform(
                KE["master_bus"]["compressor"]["makeup_gain_db"][0],
                KE["master_bus"]["compressor"]["makeup_gain_db"][1],
            )
            master_params[j, 24] = random.uniform(
                KE["master_bus"]["fader"]["gain_db"][0],
                KE["master_bus"]["fader"]["gain_db"][1],
            )

    if mix_console.num_track_control_params == 2:
        mix_params[:, :, 1] = pan_params[:, :, 0]
        mix_params = mix_params.type_as(tracks)
        track_param_dict = {
            "input_gain": {
                "gain_db": mix_params[:, :, 0],  # bs, num_tracks, 1
            },
            "stereo_panner": {
                "pan": mix_params[:, :, 1],  # bs, num_tracks, 1
            },
        }

    # applying knowledge engineered compressor and EQ
    elif mix_console.num_track_control_params != 2:
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
        mix_params[:, :, 26] = fx_send_params[:, :, 0]
        mix_params = mix_params.type_as(tracks)
        # fx_send_params= fx_send_params.type_as(tracks)
        track_param_dict = {
            "fader": {
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
            "fx_bus": {"send_db": mix_params[..., 26]},
            "stereo_panner": {
                "pan": mix_params[..., 25],
            },
        }
        fx_params = fx_params.type_as(tracks)
        fx_bus_param_dict = {
            "reverberation": {
                "band0_gain": fx_params[..., 0],
                "band1_gain": fx_params[..., 1],
                "band2_gain": fx_params[..., 2],
                "band3_gain": fx_params[..., 3],
                "band4_gain": fx_params[..., 4],
                "band5_gain": fx_params[..., 5],
                "band6_gain": fx_params[..., 6],
                "band7_gain": fx_params[..., 7],
                "band8_gain": fx_params[..., 8],
                "band9_gain": fx_params[..., 9],
                "band10_gain": fx_params[..., 10],
                "band11_gain": fx_params[..., 11],
                "band0_decay": fx_params[..., 12],
                "band1_decay": fx_params[..., 13],
                "band2_decay": fx_params[..., 14],
                "band3_decay": fx_params[..., 15],
                "band4_decay": fx_params[..., 16],
                "band5_decay": fx_params[..., 17],
                "band6_decay": fx_params[..., 18],
                "band7_decay": fx_params[..., 19],
                "band8_decay": fx_params[..., 20],
                "band9_decay": fx_params[..., 21],
                "band10_decay": fx_params[..., 22],
                "band11_decay": fx_params[..., 23],
                "mix": fx_params[..., 24],
            }
        }
        master_params = master_params.type_as(tracks)
        master_bus_param_dict = {
            "parametric_eq": {
                "low_shelf_gain_db": master_params[..., 0],
                "low_shelf_cutoff_freq": master_params[..., 1],
                "low_shelf_q_factor": master_params[..., 2],
                "band0_gain_db": master_params[..., 3],
                "band0_cutoff_freq": master_params[..., 4],
                "band0_q_factor": master_params[..., 5],
                "band1_gain_db": master_params[..., 6],
                "band1_cutoff_freq": master_params[..., 7],
                "band1_q_factor": master_params[..., 8],
                "band2_gain_db": master_params[..., 9],
                "band2_cutoff_freq": master_params[..., 10],
                "band2_q_factor": master_params[..., 11],
                "band3_gain_db": master_params[..., 12],
                "band3_cutoff_freq": master_params[..., 13],
                "band3_q_factor": master_params[..., 14],
                "high_shelf_gain_db": master_params[..., 15],
                "high_shelf_cutoff_freq": master_params[..., 16],
                "high_shelf_q_factor": master_params[..., 17],
            },
            # release and attack time must be the same
            "compressor": {
                "threshold_db": master_params[..., 18],
                "ratio": master_params[..., 19],
                "attack_ms": master_params[..., 20],
                "release_ms": master_params[..., 21],
                "knee_db": master_params[..., 22],
                "makeup_gain_db": master_params[..., 23],
            },
            "fader": {"gain_db": master_params[..., 24]},
        }

    # check param_dict for out of range parameters
    for effect_name, effect_param_dict in track_param_dict.items():
        for param_name, param_val in effect_param_dict.items():
            if param_val.min() < mix_console.param_ranges[effect_name][param_name][0]:
                print(f"{param_name} out of range {param_val.min()}")
                print(mix_console.param_ranges[effect_name][param_name][0])
            if param_val.max() > mix_console.param_ranges[effect_name][param_name][1]:
                mix_console.param_ranges[effect_name][param_name][1]
                print(
                    f"{param_name} = out of range.  ({param_val.min()},{param_val.max()})"
                )

    mixed_tracks, mix = mix_console.forward_mix_console(
        tracks,
        track_param_dict,
        fx_bus_param_dict,
        master_bus_param_dict,
        use_track_gain,
        use_track_eq,
        use_track_compressor,
        use_track_panner,
        use_fx_bus,
        use_master_bus,
    )
    # peak normalize the mix
    # mix /= mix.abs().max().clamp(min=1e-8)

    # return mix, param_dict
    # remove warmup samples
    mix = mix[..., warmup:]
    mixed_tracks = mixed_tracks[..., warmup:]
    # normalize mix
    gain_lin = 1 / mix.abs().max().clamp(min=1e-8)
    mix *= gain_lin
    mixed_tracks *= gain_lin

    return mixed_tracks, mix, track_param_dict, fx_bus_param_dict, master_bus_param_dict


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

#         # naive_mix, param_dict = naive_random_mix(track, mix_console)
#         #print("naive_mix", naive_mix.size())
#         mix, param_dict = knowledge_engineering_mix(track, mix_console, instrument_id, stereo)
#         #if mix contains nan, then print
#         if torch.isnan(mix).any():
#             print("nan")
#             print(param_dict)
#         # track = track.view(batch_size,num_tracks,1, seq_len)
#         # #print("track", track.size())


#         # sum_mix =  torch.sum(track, dim=1)
#         # # print("mix", mix.size())
#         # #print("sum_mix", sum_mix.size())

#         # if not os.path.exists("mix_KE_adv"):
#         #     os.mkdir("mix_KE_adv")
#         # save_dir = "mix_KE_adv/"

#         #export audio
#         # for j in range(batch_size):
#         #     torchaudio.save(os.path.join(save_dir,"mix_"+str(j)+".wav"), mix[j], 44100)
#         #     torchaudio.save(os.path.join(save_dir,"sum"+str(j)+".wav"), sum_mix[j], 44100)
#         #     torchaudio.save(os.path.join(save_dir,"naive"+str(j)+".wav"), naive_mix[j], 44100)
#         # print("mix", mix.size())
#         # if i==0:
#         #     break
