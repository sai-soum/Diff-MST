import os
import torch
import torchaudio

from mst.mixing import naive_random_mix
from mst.modules import BasicMixConsole, AdvancedMixConsole
from mst.dataloaders.medley import MedleyDBDataModule

datamodule = MedleyDBDataModule(
    ["/import/c4dm-datasets/MedleyDB_V1/V1", "/import/c4dm-datasets/MedleyDB_V1/V2"],
    524288,
    4,
    16,
    1,
    1,
    0.5,
    0.5,
)
datamodule.setup("fit")

root = "./"
os.makedirs(f"{root}/debug", exist_ok=True)

train_loader = datamodule.train_dataloader()
mix_fn = naive_random_mix

generate_mix_console = AdvancedMixConsole(44100, min_gain_db=-24.0, max_gain_db=24.0)

for idx, batch in enumerate(train_loader):
    tracks, instrument_id, stereo_info = batch

    sum_mix = tracks.sum(dim=1)
    # create a random mix (on GPU, if applicable)
    (
        mixed_tracks,
        mix,
        track_param_dict,
        fx_bus_param_dict,
        master_bus_param_dict,
    ) = mix_fn(tracks, generate_mix_console)

    print(mix.shape, sum_mix.shape)
    mix = mix.view(2, -1)

    mix /= mix.abs().max()
    sum_mix /= sum_mix.abs().max()

    torchaudio.save(f"{root}/debug/{idx}_ref_mix.wav", mix.cpu(), 44100)
    torchaudio.save(f"{root}/debug/{idx}_sum_mix.wav", sum_mix[0:1].cpu(), 44100)
    if idx > 25:
        break
