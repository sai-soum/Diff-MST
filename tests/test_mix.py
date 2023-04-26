import torch
from torchvision.models.resnet import resnet18
from mst.modules import (
    MixStyleTransferModel,
    SpectrogramResNetEncoder,
    TransformerController,
    BasicMixConsole,
    AdvancedMixConsole,
)
from tqdm import tqdm
from mst.mixing import naive_random_mix, knowledge_engineering_mix

sample_rate = 44100
embed_dim = 128
num_control_params = 26


mix_console = AdvancedMixConsole(sample_rate)

for n in tqdm(range(100)):
    bs = 8
    num_tracks = 4
    seq_len = 262144

    tracks = torch.randn(bs, num_tracks, seq_len) * 0.1

    mix, params = naive_random_mix(tracks, mix_console)

    if torch.isnan(mix).any():
        print("NAN")
        print(mix.shape)
        print(torch.isnan(mix).any())

        break
