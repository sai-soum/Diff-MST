import torch
from mst.baseline import (
    load_mixing_style_transfer_model,
    run_mixing_style_transfer_model,
)


if __name__ == "__main__":

    bs = 1
    tracks = 8
    seq_len = 262144

    tracks = torch.randn(bs, tracks, seq_len)
    ref_mix = torch.rand(bs, 2, seq_len)

    model = load_mixing_style_transfer_model()
    result = run_mixing_style_transfer_model(tracks, ref_mix, model, None)
    (
        pred_mix,
        pred_track_param_dict,
        pred_fx_bus_param_dict,
        pred_master_bus_param_dict,
    ) = result

    print(pred_mix.shape)
