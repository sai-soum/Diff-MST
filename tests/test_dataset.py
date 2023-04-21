import os
import torch
import torchaudio

from mst.mixing import naive_random_mix
from mst.modules import BasicMixConsole
from mst.dataloaders.medley import MedleyDBDataModule

datamodule = MedleyDBDataModule(
    ["/scratch/csteinmetz1/V1", "/scratch/csteinmetz1/MedleyDB_V2"],
    524288,
    4,
    16,
    0.1,
    0.1,
)
datamodule.setup("fit")

root = "/fsx/home-csteinmetz1/tmp"
os.makedirs(f"{root}/debug", exist_ok=True)

train_loader = datamodule.train_dataloader()
mix_fn = naive_random_mix
generate_mix_console = BasicMixConsole(44100, min_gain_db=-24.0, max_gain_db=24.0)

for idx, batch in enumerate(train_loader):
    tracks = batch

    sum_mix = tracks.sum(dim=1)
    # create a random mix (on GPU, if applicable)
    ref_mix, ref_param_dict = mix_fn(tracks, generate_mix_console)

    torchaudio.save(f"{root}/debug/ref_mix_{idx}.wav", ref_mix[0].cpu(), 44100)
    torchaudio.save(f"{root}/debug/sum_mix_{idx}.wav", sum_mix[0:1].cpu(), 44100)
    if idx > 10:
        break
