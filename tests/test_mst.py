import torch
from torchvision.models.resnet import resnet18
from mst.modules import (
    MixStyleTransferModel,
    SpectrogramResNetEncoder,
    TransformerController,
    BasicMixConsole,
    AdvancedMixConsole,
)

sample_rate = 44100
embed_dim = 128
num_control_params = 10

track_encoder = SpectrogramResNetEncoder()
mix_encoder = SpectrogramResNetEncoder()
controller = TransformerController(
    embed_dim=embed_dim, num_control_params=num_control_params
)


mix_console = BasicMixConsole(sample_rate)

model = MixStyleTransferModel(track_encoder, mix_encoder, controller, mix_console)

bs = 8
num_tracks = 4
seq_len = 262144

tracks = torch.randn(bs, num_tracks, seq_len)
ref_mix = torch.randn(bs, 2, seq_len)

model(tracks, ref_mix)
