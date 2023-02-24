import torch
from torchvision.models.resnet import resnet18
from mst.modules import (
    MixStyleTransferModel,
    SpectrogramEncoder,
    TransformerController,
    BasicMixConsole,
    AdvancedMixConsole,
)

sample_rate = 44100
embed_dim = 128
num_control_params = 10

track_encoder = SpectrogramEncoder(
    sample_rate,
    resnet18(num_classes=embed_dim),
    embed_dim=embed_dim,
)
mix_encoder = SpectrogramEncoder(
    sample_rate,
    resnet18(num_classes=embed_dim),
    embed_dim=embed_dim,
)
controller = TransformerController(
    embed_dim=embed_dim, num_control_params=num_control_params
)


param_ranges = {
    "input_gain": {
        "gain_db": [-80.0, 24.0],
    },
    "stereo_panner": {
        "pan": [0.0, 1.0],
    },
}
mix_console = BasicMixConsole(sample_rate, param_ranges)

model = MixStyleTransferModel(track_encoder, mix_encoder, controller, mix_console)

bs = 8
num_tracks = 4
seq_len = 262144

tracks = torch.randn(bs, num_tracks, 1, seq_len)
ref_mix = torch.randn(bs, 2, seq_len)

model(tracks, ref_mix)
