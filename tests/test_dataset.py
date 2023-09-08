import os
import torch
import torchaudio

from mst.mixing import naive_random_mix
from mst.modules import AdvancedMixConsole
from mst.dataloader import MultitrackDataModule

datamodule = MultitrackDataModule(
    root_dirs=["/import/c4dm-datasets-ext/mixing-secrets/", "/import/c4dm-datasets/"],
    metadata_files=["data/cambridge.yaml", "data/medley.yaml"],
    length=262144,
    min_tracks=8,
    max_tracks=8,
    batch_size=2,
    num_workers=4,
    num_train_passes=20,
    num_val_passes=1,
    train_buffer_size_gb=0.1,
    val_buffer_size_gb=0.1,
    target_track_lufs_db=-48.0,
)
datamodule.setup("fit")

root = "./"
os.makedirs(f"{root}/debug", exist_ok=True)

train_loader = datamodule.train_dataloader()
mix_fn = naive_random_mix
use_gpu = True

generate_mix_console = AdvancedMixConsole(44100)

for idx, batch in enumerate(train_loader):
    tracks, instrument_id, stereo_info = batch

    if use_gpu:
        tracks = tracks.cuda()

    sum_mix = tracks.sum(dim=1)
    # create a random mix (on GPU, if applicable)
    (
        mixed_tracks,
        mix,
        track_param_dict,
        fx_bus_param_dict,
        master_bus_param_dict,
    ) = mix_fn(
        tracks,
        generate_mix_console,
        use_track_input_fader=False,
        use_output_fader=False,
        use_fx_bus=True,
        use_master_bus=True,
    )

    mix = mix[0, ...]
    sum_mix = sum_mix[0, ...]
    print(mix.shape, sum_mix.shape)
    print(mix.abs().max(), sum_mix.abs().max())
    mix = mix.view(2, -1)
    sum_mix = sum_mix.repeat(2, 1)

    mix /= mix.abs().max()
    sum_mix /= sum_mix.abs().max()

    # split mix into a and b sections
    mix_a = mix[:, 0 : mix.shape[1] // 2]
    mix_b = mix[:, mix.shape[1] // 2 :]

    torchaudio.save(f"{root}/debug/{idx}_ref_mix_a.wav", mix_a.cpu(), 44100)
    torchaudio.save(f"{root}/debug/{idx}_ref_mix_b.wav", mix_b.cpu(), 44100)
    torchaudio.save(f"{root}/debug/{idx}_sum_mix.wav", sum_mix.cpu(), 44100)
    if idx > 25:
        break
