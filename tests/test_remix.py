import torch
import torchaudio
from mst.modules import Remixer, AdvancedMixConsole
from mst.dataloader import MixDataset

if __name__ == "__main__":
    root_dir = "/import/c4dm-datasets-ext/mtg-jamendo"
    mix_dataset = MixDataset(root_dir, length=262144)
    mix_dataloader = torch.utils.data.DataLoader(
        mix_dataset, batch_size=4, num_workers=4
    )

    for batch_idx, batch in enumerate(mix_dataloader):
        mix, label = batch

        for i in range(mix.shape[0]):
            torchaudio.save(f"debug/{batch_idx}-{i}-{label[i]}.wav", mix[i], 44100)

        print(mix.shape, label)
        if batch_idx > 10:
            break
