import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

from mst.modules import (
    MixStyleTransferModel,
    SpectrogramResNetEncoder,
    TransformerController,
    BasicMixConsole,
    AdvancedMixConsole,
)

if False:
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
    model.cuda()

    bs = 1
    num_tracks = 8
    seq_len = 262144

    tracks = torch.randn(bs, num_tracks, seq_len)
    ref_mix = torch.randn(bs, 2, seq_len)

    tracks = tracks.cuda()
    ref_mix = ref_mix.cuda()

    with profile(
        activities=[
            ProfilerActivity.CUDA,
            ProfilerActivity.CPU,
        ],
        profile_memory=True,
        record_shapes=True,
        with_modules=False,
        with_stack=True,
    ) as prof:
        (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        ) = model(tracks, ref_mix)

    print(
        prof.key_averages(group_by_stack_n=5).table(
            sort_by="self_cuda_memory_usage", row_limit=25
        )
    )
