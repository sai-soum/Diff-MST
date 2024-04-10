# evaluation script using data from the dataloader (test set)
import os
import json
import torch
import torchaudio
import pyloudnorm as pyln
from mst.loss import AudioFeatureLoss
from mst.utils import load_diffmst, run_diffmst
from mst.baseline import (
    equal_loudness_mix,
    load_mixing_style_transfer_model,
    run_mixing_style_transfer_model,
)
from mst.dataloader import MultitrackDataModule

from frechet_audio_distance import FrechetAudioDistance


def fad_score(
    fad: FrechetAudioDistance,
    audio_eval: torch.Tensor,
    audio_background: torch.Tensor,
    sample_rate: int,
):
    embds_background = fad.get_embeddings(
        [audio_background.squeeze().view(-1).numpy()],
        sr=sample_rate,
    )
    embds_eval = fad.get_embeddings(
        [audio_eval.squeeze().view(-1).numpy()],
        sr=sample_rate,
    )

    # Compute statistics and FAD score
    mu_background, sigma_background = fad.calculate_embd_statistics(embds_background)
    mu_eval, sigma_eval = fad.calculate_embd_statistics(embds_eval)

    fad_score = fad.calculate_frechet_distance(
        mu_background, sigma_background, mu_eval, sigma_eval
    )

    return float(fad_score)


if __name__ == "__main__":

    root_dir = "/import/c4dm-datasets-ext/Diff-MST/eval"
    num_examples = 1000
    meter = pyln.Meter(44100)
    target_lufs_db = -22.0
    output_dir = os.path.join(root_dir, "generate")
    os.makedirs(output_dir, exist_ok=True)

    methods = {
        "mst": {
            "model": load_mixing_style_transfer_model(),
            "func": run_mixing_style_transfer_model,
        },
        "diffmst-16": {
            "model": load_diffmst(
                "/import/c4dm-datasets-ext/Diff-MST/DiffMST/b4naquji/config.yaml",
                "/import/c4dm-datasets-ext/Diff-MST/DiffMST/b4naquji/checkpoints/epoch=191-step=626608.ckpt",
            ),
            "func": run_diffmst,
        },
        "diffmst-stft-8": {
            "model": load_diffmst(
                "configs/models/gain+eq+comp-feat.yaml",
                "/import/c4dm-datasets-ext/diffmst_logs_soum/DiffMST/u4x49p4g/checkpoints/epoch=184-step=1850000.ckpt",
            ),
            "func": run_diffmst,
        },
        "diffmst-stft-16": {
            "model": load_diffmst(
                "configs/models/gain+eq+comp-feat.yaml",
                "/import/c4dm-datasets-ext/diffmst_logs_soum/DiffMST/80f8mo8c/checkpoints/epoch=115-step=1160000.ckpt",
            ),
            "func": run_diffmst,
        },
        "diffmst-stft+AF-8": {
            "model": load_diffmst(
                "configs/models/gain+eq+comp-feat.yaml",
                "/import/c4dm-datasets-ext/diffmst_logs_soum/DiffMST/shq4oguz/checkpoints/epoch=185-step=1860000.ckpt",
            ),
            "func": run_diffmst,
        },
        "diffmst-stft+AF-16": {
            "model": load_diffmst(
                "configs/models/gain+eq+comp-feat.yaml",
                "/import/c4dm-datasets-ext/diffmst_logs_soum/DiffMST/hh65jlmt/checkpoints/epoch=117-step=1180000.ckpt",
            ),
            "func": run_diffmst,
        },
        "equal_loudness": {
            "model": (None, None),
            "func": equal_loudness_mix,
        },
    }

    # setup the mix style metric
    audio_feature_loss = AudioFeatureLoss(
        weights=[
            0.1,  # rms
            0.001,  # crest factor
            1.0,  # stereo width
            1.0,  # stereo imbalance
            0.1,  # bark spectrum
        ],
        sample_rate=44100,
        use_clap=False,
    )

    frechet = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=16000,
        use_pca=False,
        use_activation=False,
        verbose=False,
    )

    # setup the datamodule
    dm = MultitrackDataModule(
        track_root_dirs=[
            "/import/c4dm-datasets-ext/mixing-secrets/",
            "/import/c4dm-datasets/",
        ],
        mix_root_dirs=["/import/c4dm-datasets/MUSDB18HQ/test/"],
        metadata_files=["./data/cambridge.yaml", "./data/medley.yaml"],
        length=262144,
        min_tracks=2,
        max_tracks=16,
        batch_size=1,
        num_workers=1,
        num_train_passes=20,
        num_val_passes=1,
        train_buffer_size_gb=4.0,
        val_buffer_size_gb=0.2,
        target_track_lufs_db=-48.0,
        randomize_ref_mix_gain=True,
    )

    dataloader = dm.val_dataloader()

    results = {}

    for method_name, method in methods.items():
        results[method_name] = {}

    # now iterate over the test dataloader
    for bidx, batch in enumerate(dataloader):

        if bidx > num_examples:
            break

        tracks, stereo_info, track_metadata, track_padding, ref_mix, song_name = batch

        print(bidx, song_name)

        # loudness normalize ref to -16 dB LUFS
        loudness = meter.integrated_loudness(ref_mix.squeeze(0).permute(1, 0).numpy())
        ref_mix = ref_mix * 10 ** ((-16.0 - loudness) / 20)

        # save the reference track
        ref_filepath = os.path.join(
            output_dir,
            f"{bidx:03d}-ref.wav",
        )
        torchaudio.save(ref_filepath, ref_mix.view(2, -1), 44100)

        # mix with each method
        for method_name, method in methods.items():
            print(method_name)
            # tracks (torch.Tensor): Set of input tracks with shape (bs, num_tracks, seq_len)
            # ref_audio (torch.Tensor): Reference mix with shape (bs, 2, seq_len)

            models = method["model"]
            func = method["func"]

            print(tracks.shape, ref_mix.shape)

            with torch.no_grad():
                result = func(
                    tracks.clone(),
                    ref_mix.clone(),
                    models,
                    track_start_idx=0,
                    ref_start_idx=0,
                )

                (
                    pred_mix,
                    pred_track_param_dict,
                    pred_fx_bus_param_dict,
                    pred_master_bus_param_dict,
                ) = result

            bs, chs, seq_len = pred_mix.shape

            # measure audio feature loss
            loss_dict = audio_feature_loss(pred_mix, ref_mix)

            for loss_key, loss_val in loss_dict.items():
                if loss_key not in results[method_name]:
                    results[method_name][loss_key] = []
                results[method_name][loss_key].append(loss_val.item())

            # measure FAD
            fad = fad_score(frechet, pred_mix, ref_mix, 44100)
            print(fad)

            if "fad" not in results[method_name]:
                results[method_name]["fad"] = []
            results[method_name]["fad"].append(fad)

            # loudness normalize the output mix
            mix_lufs_db = meter.integrated_loudness(
                pred_mix.squeeze(0).permute(1, 0).numpy()
            )
            print(mix_lufs_db)
            lufs_delta_db = target_lufs_db - mix_lufs_db
            pred_mix = pred_mix * 10 ** (lufs_delta_db / 20)

            # save resulting audio and parameters
            mix_filepath = os.path.join(
                output_dir,
                f"{bidx:03d}-{method_name}.wav",
            )
            torchaudio.save(mix_filepath, pred_mix.view(chs, -1), 44100)

            # save results to json
            json_filepath = os.path.join("outputs", "eval_generate_results.json")
            with open(json_filepath, "w") as fp:
                json.dump(results, fp, indent=4)
