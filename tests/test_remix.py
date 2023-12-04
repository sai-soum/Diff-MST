import torch
from mst.modules import Remixer, AdvancedMixConsole
from mst.dataloader import MixDataset

if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/mtg-jamendo"
    mix_dataset = MixDataset(root_dir, length=262144)
    mix_dataloader = torch.utils.data.DataLoader(mix_dataset, batch_size=4)

    mix_console = AdvancedMixConsole(44100)
    remixer = Remixer(44100)

    remixer.cuda()

    for batch_idx, batch in enumerate(mix_dataloader):
        mix = batch

        mix = mix.cuda()

        # create remix
        remix, track_params, fx_bus_params, master_bus_params = remixer(
            mix, mix_console
        )

        print(batch_idx, mix.abs().max(), remix.abs().max())
