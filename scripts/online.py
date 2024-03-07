import os
import glob
import torch
import argparse
import torchaudio
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from tqdm import tqdm

from mst.loss import AudioFeatureLoss, StereoCLAPLoss, compute_barkspectrum
from mst.modules import AdvancedMixConsole


def optimize(
    tracks: torch.Tensor,
    ref_mix: torch.Tensor,
    mix_console: torch.nn.Module,
    loss_function: torch.nn.Module,
    init_scale: float = 0.001,
    lr: float = 1e-3,
    n_iters: int = 100,
):
    """Create a mix from the tracks that is as close as possible to the reference mixture.

    Args:
        tracks (torch.Tensor): Tensor of shape (n_tracks, n_samples).
        ref_mix (torch.Tensor): Tensor of shape (2, n_samples).
        mix_console (torch.nn.Module): Mix console instance. (e.g. AdvancedMixConsole)
        loss_function (torch.nn.Module): Loss function instance. (e.g. AudioFeatureLoss)
        n_iters (int): Number of iterations for the optimization.

    Returns:
        torch.Tensor: Tensor of shape (2, n_samples) that is as close as possible to the reference mixture.
    """
    loss_history = {"loss": []}  # lists to store loss values

    # initialize the mix console parameters to optimize
    track_params = init_scale * torch.randn(
        tracks.shape[0], mix_console.num_track_control_params
    )
    fx_bus_params = init_scale * torch.randn(1, mix_console.num_fx_bus_control_params)
    master_bus_params = init_scale * torch.randn(
        1, mix_console.num_master_bus_control_params
    )

    # move parameters to same device as tracks
    track_params = track_params.type_as(tracks)
    fx_bus_params = fx_bus_params.type_as(tracks)
    master_bus_params = master_bus_params.type_as(tracks)

    # require gradients for the parameters
    track_params.requires_grad = True
    fx_bus_params.requires_grad = True
    master_bus_params.requires_grad = True

    # create optimizer and link to console parameters
    optimizer = torch.optim.Adam(
        [track_params, fx_bus_params, master_bus_params], lr=lr
    )

    pbar = tqdm(range(n_iters))

    # reshape
    tracks = tracks.unsqueeze(0)
    ref_mix = ref_mix.unsqueeze(0)
    track_params = track_params.unsqueeze(0)
    fx_bus_params = fx_bus_params.unsqueeze(0)
    master_bus_params = master_bus_params.unsqueeze(0)

    for n in pbar:
        optimizer.zero_grad()

        # mix the tracks using the mix console
        # the mix console parameters are sigmoided to ensure they are in the range [0, 1]
        result = mix_console(
            tracks,
            torch.sigmoid(track_params),
            torch.sigmoid(fx_bus_params),
            torch.sigmoid(master_bus_params),
            use_fx_bus=False,
        )
        mix = result[1]
        track_param_dict = result[2]
        fx_bus_param_dict = result[3]
        master_bus_param_dict = result[4]

        # compute loss
        loss = 0
        losses = loss_function(mix, ref_mix)
        for loss_name, loss_value in losses.items():
            loss += loss_value

        # compute gradients and update parameters
        loss.backward()
        optimizer.step()

        # update progress bar
        pbar.set_description(f"Loss: {loss.item():.4f}")

        # store loss values
        loss_history["loss"].append(loss.item())
        for loss_name, loss_value in losses.items():
            if loss_name not in loss_history:
                loss_history[loss_name] = []
            loss_history[loss_name].append(loss_value.item())

    # reshape
    mix = mix.squeeze(0)
    track_params = track_params  # .squeeze(0)
    fx_bus_params = fx_bus_params  # .squeeze(0)
    master_bus_params = master_bus_params  # .squeeze(0)

    return (
        mix,
        track_params,
        track_param_dict,
        fx_bus_params,
        fx_bus_param_dict,
        master_bus_params,
        master_bus_param_dict,
        loss_history,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track_dir",
        type=str,
        help="Path to directory with tracks.",
    )
    parser.add_argument(
        "--ref_mix",
        type=str,
        help="Path to reference mixture.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        help="Number of iterations.",
        default=250,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate.",
        default=1e-3,
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function.",
        choices=["feat", "clap"],
        default="feat",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory.",
        default="outputs",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU for optimization.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="Sample rate of tracks.",
        default=44100,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=524288,
        help="Analysis block size.",
    )
    parser.add_argument(
        "--start_time_s",
        type=float,
        default=32.0,
        help="Analysis block start time.",
    )
    parser.add_argument(
        "--target_track_lufs_db",
        type=float,
        default=-48.0,
    )
    parser.add_argument(
        "--target_mix_lufs_db",
        type=float,
        default=-14.0,
    )
    parser.add_argument(
        "--stem_separation",
        action="store_true",
    )
    parser
    args = parser.parse_args()

    meter = pyln.Meter(args.sample_rate)
    run_name = f"{os.path.basename(args.track_dir)}-->{os.path.basename(args.ref_mix).split('.')[0]}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # -------------------------- data loading -------------------------- #
    # load tracks
    tracks = []
    print(f"Loading tracks for current run: {run_name}...")
    track_filepaths = sorted(glob.glob(os.path.join(args.track_dir, "*.wav")))
    num_tracks = len(track_filepaths)
    for track_idx, track_filepath in enumerate(track_filepaths):
        track, track_sample_rate = torchaudio.load(os.path.join(track_filepath))
        print(
            f"{track_idx+1}/{num_tracks}: {track.shape} {os.path.basename(track_filepath)}"
        )

        # check if track has same sample rate as reference mixture
        # if not, resample
        if track_sample_rate != args.sample_rate:
            track = torchaudio.transforms.Resample(track_sample_rate, args.sample_rate)(
                track
            )

        # check if the track is silent
        for ch_idx in range(track.shape[0]):
            # measure loudness
            track_lufs_db = meter.integrated_loudness(track[ch_idx, :].numpy())

            if track_lufs_db < -60.0:
                print(f"Track is inactive at {track_lufs_db:0.2f} dB. Skipping...")
                continue
            else:
                # loudness normalize
                delta_lufs_db = args.target_track_lufs_db - track_lufs_db
                delta_lufs_lin = 10 ** (delta_lufs_db / 20)
                tracks.append(delta_lufs_lin * track[ch_idx, :])

    tracks = torch.stack(tracks)  # shape: (n_tracks, n_samples)

    # load reference mixture
    ref_mix, ref_sample_rate = torchaudio.load(args.ref_mix)

    if ref_sample_rate != args.sample_rate:
        ref_mix = torchaudio.transforms.Resample(ref_sample_rate, args.sample_rate)(
            ref_mix
        )
    mix_lufs_db = meter.integrated_loudness(ref_mix.permute(1, 0).numpy())
    delta_lufs_db = args.target_mix_lufs_db - mix_lufs_db
    delta_lufs_lin = 10 ** (delta_lufs_db / 20)
    ref_mix = delta_lufs_lin * ref_mix

    # use only a subsection of the reference mixture and tracks
    # this is to speed up the optimization
    start_time_s = args.start_time_s
    start_sample = int(start_time_s * args.sample_rate)

    ref_mix_section = ref_mix[:, start_sample : start_sample + args.block_size]
    tracks_section = tracks[:, start_sample : start_sample + args.block_size]

    print(ref_mix.shape, tracks.shape)
    print(ref_mix_section.shape, tracks_section.shape)

    if args.use_gpu:
        ref_mix_section = ref_mix_section.cuda()
        tracks_section = tracks_section.cuda()

    # -------------------------- setup -------------------------- #
    # mix console will use the same sample rate as the tracks
    mix_console = AdvancedMixConsole(args.sample_rate)

    weights = [
        0.1,  # rms
        0.001,  # crest factor
        1.0,  # stereo width
        1.0,  # stereo imbalance
        1.00,  # bark spectrum
        100.0,  # clap
    ]

    if args.loss == "feat":
        loss_function = AudioFeatureLoss(
            weights,
            args.sample_rate,
            stem_separation=args.stem_separation,
        )
    elif args.loss == "clap":
        loss_function = StereoCLAPLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    if args.use_gpu:
        loss_function.cuda()

    # -------------------------- optimization -------------------------- #
    result = optimize(
        tracks_section,
        ref_mix_section,
        mix_console,
        loss_function,
        n_iters=args.n_iters,
    )

    mix = result[0]
    track_params = result[1]
    track_param_dict = result[2]
    fx_bus_parms = result[3]
    fx_bus_param_dict = result[4]
    master_bus_params = result[5]
    master_bus_param_dict = result[6]
    loss_history = result[7]

    ref_mix = ref_mix.squeeze(0).cpu()
    mono_mix = tracks.sum(dim=0).repeat(2, 1).cpu()

    # print(track_param_dict)
    # print(fx_bus_param_dict)
    # print(master_bus_param_dict)
    print(mix.abs().max())
    print(mono_mix.abs().max())

    # ----------------------- full mix generation ---------------------- #
    # iterate over the tracks in blocks to mix the entire song
    # this is to avoid memory issues
    block_size = args.block_size
    n_blocks = tracks.shape[-1] // block_size
    full_mix = torch.zeros(2, tracks.shape[-1])
    for block_idx in tqdm(range(n_blocks)):
        tracks_block = tracks[:, block_idx * block_size : (block_idx + 1) * block_size]
        tracks_block = tracks_block.type_as(tracks_section)

        with torch.no_grad():
            result = mix_console(
                tracks_block.unsqueeze(0),
                torch.sigmoid(track_params),
                torch.sigmoid(fx_bus_parms),
                torch.sigmoid(master_bus_params),
                use_fx_bus=False,
            )
            mix_block = result[1].squeeze(0).cpu()
            full_mix[
                :, block_idx * block_size : (block_idx + 1) * block_size
            ] = mix_block

    # loudness normalize
    full_mix /= full_mix.abs().max()
    mono_mix /= mono_mix.abs().max()

    mono_mix_section = mono_mix[:, start_sample : start_sample + args.block_size]

    # -------------------------- analyze mixes -------------------------- #

    ref_spec = compute_barkspectrum(ref_mix.unsqueeze(0), sample_rate=args.sample_rate)
    pred_spec = compute_barkspectrum(
        full_mix.unsqueeze(0), sample_rate=args.sample_rate
    )

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    axs[0].plot(ref_spec[0, :, 0], label="ref-mid", color="tab:orange")
    axs[0].plot(pred_spec[0, :, 0], label="pred-mid", color="tab:blue")
    axs[1].plot(ref_spec[0, :, 1], label="ref-side", color="tab:orange")
    axs[1].plot(pred_spec[0, :, 1], label="pred-side", color="tab:blue")
    axs[0].legend()
    axs[1].legend()
    plt.savefig(os.path.join(output_dir, "plots", "bark_specta.png"))
    plt.close("all")

    for idx, (loss_name, loss_vals) in enumerate(loss_history.items()):
        fig, ax = plt.subplots(1, 1)
        ax.plot(loss_vals, label=loss_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"{loss_name}")
        plt.savefig(os.path.join(output_dir, "plots", f"{loss_name}.png"))
        plt.close("all")

    # -------------------------- save results -------------------------- #
    # save mix
    torchaudio.save(
        os.path.join(output_dir, "pred_mix_section.wav"),
        mix.squeeze(0).cpu(),
        args.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "ref_mix_section.wav"),
        ref_mix_section.cpu(),
        args.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "mono_mix_section.wav"),
        mono_mix_section,
        args.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "pred_mix.wav"),
        full_mix,
        args.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "mono_mix.wav"),
        mono_mix,
        args.sample_rate,
    )
    torchaudio.save(
        os.path.join(output_dir, "ref_mix.wav"),
        ref_mix,
        args.sample_rate,
    )
