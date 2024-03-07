import torch
import torchaudio

from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from mst.modules import AdvancedMixConsole


class Remixer(torch.nn.Module):
    def __init__(self, sample_rate: int) -> None:
        super().__init__()
        self.sample_rate = sample_rate

        # load source separation model
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.stem_separator = bundle.get_model()
        self.stem_separator.eval()
        # get sources list
        self.sources_list = list(self.stem_separator.sources)

        # load mix console
        self.mix_console = AdvancedMixConsole(sample_rate)

    def forward(self, x: torch.Tensor):
        """Take a tensor of mixes, separate, and then remix.

        Args:
            x (torch.Tensor): Tensor of mixes with shape (batch, 2, samples)

        Returns:
            remix (torch.Tensor): Tensor of remixes with shape (batch, 2, samples)
            sum_mix (torch.Tensor): Tensor of mixes from separeted outputs with shape (batch, 2, samples)
            track_params (torch.Tensor): Tensor of track params with shape (batch, 8, num_track_control_params)
            fx_bus_params (torch.Tensor): Tensor of fx bus params with shape (batch, num_fx_bus_control_params)
            master_bus_params (torch.Tensor): Tensor of master bus params with shape (batch, num_master_bus_control_params)
        """
        bs, chs, seq_len = x.size()

        # separate
        sources = self.stem_separator(x)  # bs, 4, 2, seq_len
        sum_mix = sources.sum(dim=1)  # bs, 2, seq_len

        # convert sources to mono tracks
        tracks = sources.view(bs, 8, -1)

        # provide some headroom before mixing
        tracks *= 10 ** (-32.0 / 20.0)

        # generate random mix parameters
        track_params = torch.rand(bs, 8, self.mix_console.num_track_control_params)
        fx_bus_params = torch.rand(bs, self.mix_console.num_fx_bus_control_params)
        master_bus_params = torch.rand(
            bs, self.mix_console.num_master_bus_control_params
        )

        # the forward expects params in range of (0,1)
        result = self.mix_console(
            tracks,
            track_params,
            fx_bus_params,
            master_bus_params,
        )

        # get the remix
        remix = result[1]

        return remix, sum_mix, track_params, fx_bus_params, master_bus_params


if __name__ == "__main__":
    # create the remixer
    remixer = Remixer(44100)

    # get a mix
    mix, sample_rate = torchaudio.load("outputs/output/ref_mix.wav")

    mix = mix.unsqueeze(0)

    # remix
    remix, sum_mix = remixer(mix)

    # peak normalize
    remix = remix / remix.abs().max()

    torchaudio.save("outputs/output/remix.wav", remix.squeeze(0), sample_rate=44100)
    torchaudio.save(
        "outputs/output/separated_sum_mix.wav", sum_mix.squeeze(0), sample_rate=44100
    )
