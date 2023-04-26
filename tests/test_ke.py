import os
import torch
import torchaudio

from mst.mixing import knowledge_engineering_mix
from mst.modules import BasicMixConsole, AdvancedMixConsole
from mst.dataloaders.medley import MedleyDBDataset

dataset = MedleyDBDataset(root_dirs=["/scratch/medleydb/V1"], subset="train")
print(len(dataset))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# mix_console = mst.modules.BasicMixConsole(sample_rate=44100.0)
mix_console = AdvancedMixConsole(sample_rate=44100.0)
for i, (track, instrument_id, stereo) in enumerate(dataloader):
    print("\n\n mixing")
    print("track", track.size())
    batch_size, num_tracks, seq_len = track.size()

    print(instrument_id)
    print(stereo)

    mix, param_dict = knowledge_engineering_mix(
        track, mix_console, instrument_id, stereo
    )
    print(param_dict)
    sum_mix = torch.sum(track, dim=1)
    print("mix", mix.size())

    save_dir = "debug"
    os.makedirs(save_dir, exist_ok=True)

    # export audio
    for j in range(batch_size):
        torchaudio.save(os.path.join(save_dir, "mix_" + str(j) + ".wav"), mix[j], 44100)
        torchaudio.save(
            os.path.join(save_dir, "sum" + str(j) + ".wav"), sum_mix[j], 44100
        )

    print("mix", mix.size())
    if i == 0:
        break
