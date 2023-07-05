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
num_track_control_params = 27
num_fx_bus_control_params = 25
num_master_bus_control_params = 24
use_fx_bus = True
use_master_bus = True

track_encoder = SpectrogramResNetEncoder()
mix_encoder = SpectrogramResNetEncoder()
controller = TransformerController(
    embed_dim=embed_dim,
    num_track_control_params=num_track_control_params,
    num_fx_bus_control_params=num_fx_bus_control_params,
    num_master_bus_control_params=num_master_bus_control_params,
)


mix_console = AdvancedMixConsole(sample_rate)

model = MixStyleTransferModel(track_encoder, mix_encoder, controller, mix_console)

bs = 8
num_tracks = 4
seq_len = 262144

tracks = torch.randn(bs, num_tracks, seq_len)
ref_mix = torch.randn(bs, 2, seq_len)

(
    mixed_tracks,
    mix,
    track_param_dict,
    fx_bus_param_dict,
    master_bus_param_dict,
) = model(tracks, ref_mix)
print(mix.shape)
print(mix)
