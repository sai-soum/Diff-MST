import math
import torch
from typing import Callable, Optional, List
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from mst.panns import Cnn14

from dasp_pytorch.functional import (
    gain,
    stereo_panner,
    compressor,
    parametric_eq,
    stereo_bus,
    noise_shaped_reverberation,
)


class MixStyleTransferModel(torch.nn.Module):
    def __init__(
        self,
        track_encoder: torch.nn.Module,
        mix_encoder: torch.nn.Module,
        controller: torch.nn.Module,
        sum_and_diff: bool = False,
    ) -> None:
        super().__init__()
        self.track_encoder = track_encoder
        self.mix_encoder = mix_encoder
        self.controller = controller
        self.sum_and_diff = sum_and_diff

    def forward(
        self,
        tracks: torch.torch.Tensor,
        ref_mix: torch.torch.Tensor,
        track_padding_mask: Optional[torch.Tensor] = None,
    ):
        bs, num_tracks, seq_len = tracks.size()

        # first process the tracks
        track_embeds = self.track_encoder(tracks.view(bs * num_tracks, 1, -1))
        track_embeds = track_embeds.view(bs, num_tracks, -1)  # restore

        # compute mid/side from the reference mix
        if self.sum_and_diff:
            ref_mix_mid = ref_mix.sum(dim=1)
            ref_mix_side = ref_mix[..., 0:1, :] - ref_mix[..., 1:2, :]

            # process the reference mix

            mid_embeds = self.mix_encoder(ref_mix_mid)
            side_embeds = self.mix_encoder(ref_mix_side)
            mix_embeds = torch.stack((mid_embeds, side_embeds), dim=1)
        else:
            mix_embeds = self.mix_encoder(ref_mix.view(bs * 2, 1, -1))
            mix_embeds = mix_embeds.view(bs, 2, -1)  # restore

        # controller will predict mix parameters for each stem based on embeds
        track_params, fx_bus_params, master_bus_params = self.controller(
            track_embeds,
            mix_embeds,
            track_padding_mask,
        )

        return (
            track_params,
            fx_bus_params,
            master_bus_params,
        )


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


def denormalize_parameters(param_dict: dict, param_ranges: dict):
    """Given parameters on (0,1) restore them to the ranges expected by the effect."""
    denorm_param_dict = {}
    for effect_name, effect_param_dict in param_dict.items():
        denorm_param_dict[effect_name] = {}
        for param_name, param_tensor in effect_param_dict.items():
            # check for out of range parameters
            if param_tensor.min() < 0 or param_tensor.max() > 1:
                raise ValueError(
                    f"Parameter {param_name} of effect {effect_name} is out of range."
                )

            param_val_denorm = denormalize(
                param_tensor,
                param_ranges[effect_name][param_name][1],
                param_ranges[effect_name][param_name][0],
            )
            denorm_param_dict[effect_name][param_name] = param_val_denorm
    return denorm_param_dict


class AdvancedMixConsole(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        input_min_gain_db: float = -48.0,
        input_max_gain_db: float = 48.0,
        output_min_gain_db: float = -48.0,
        output_max_gain_db: float = 48.0,
        min_send_db: float = -80.0,
        max_send_db: float = +12.0,
        eq_min_gain_db: float = -12.0,
        eq_max_gain_db: float = 12.0,
        min_pan: float = 0.0,
        max_pan: float = 1.0,
        reverb_min_band_gain: float = 0.0,
        reverb_max_band_gain: float = 1.0,
        reverb_min_band_decay: float = 0.0,
        reverb_max_band_decay: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            "input_fader": {"gain_db": (input_min_gain_db, input_max_gain_db)},
            "output_fader": {"gain_db": (output_min_gain_db, output_max_gain_db)},
            "parametric_eq": {
                "low_shelf_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "low_shelf_cutoff_freq": (20, 2000),
                "low_shelf_q_factor": (0.1, 5.0),
                "band0_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band0_cutoff_freq": (80, 2000),
                "band0_q_factor": (0.1, 5.0),
                "band1_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band1_cutoff_freq": (2000, 8000),
                "band1_q_factor": (0.1, 5.0),
                "band2_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band2_cutoff_freq": (8000, 12000),
                "band2_q_factor": (0.1, 5.0),
                "band3_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "band3_cutoff_freq": (12000, (sample_rate // 2) - 1000),
                "band3_q_factor": (0.1, 5.0),
                "high_shelf_gain_db": (eq_min_gain_db, eq_max_gain_db),
                "high_shelf_cutoff_freq": (6000, (sample_rate // 2) - 1000),
                "high_shelf_q_factor": (0.1, 5.0),
            },
            "compressor": {
                "threshold_db": (-60.0, 0.0),
                "ratio": (1.0, 10.0),
                "attack_ms": (5.0, 250.0),
                "release_ms": (10.0, 250.0),
                "knee_db": (3.0, 12.0),
                "makeup_gain_db": (0.0, 6.0),
            },
            "reverberation": {
                "band0_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band1_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band2_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band3_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band4_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band5_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band6_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band7_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band8_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band9_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band10_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band11_gain": (reverb_min_band_gain, reverb_max_band_gain),
                "band0_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band1_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band2_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band3_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band4_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band5_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band6_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band7_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band8_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band9_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band10_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "band11_decay": (reverb_min_band_decay, reverb_max_band_decay),
                "mix": (0.0, 1.0),
            },
            "fx_bus": {"send_db": (min_send_db, max_send_db)},
            "stereo_panner": {"pan": (min_pan, max_pan)},
        }
        self.num_track_control_params = 27
        self.num_fx_bus_control_params = 25
        self.num_master_bus_control_params = 26

    def forward_mix_console(
        self,
        tracks: torch.torch.Tensor,
        track_param_dict: dict,
        fx_bus_param_dict: dict,
        master_bus_param_dict: dict,
        use_track_input_fader: bool = True,
        use_track_eq: bool = True,
        use_track_compressor: bool = True,
        use_track_panner: bool = True,
        use_fx_bus: bool = True,
        use_master_bus: bool = True,
        use_output_fader: bool = True,
    ):
        """

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            track_param_dict (dict): Denormalized parameter values for the gain, eq, compressor, and panner
            fx_bus_param_dict (dict): Denormalized parameter values for the fx bus
            master_bus_param_dict (dict): Denormalized parameter values for the master bus
            use_track_input_fader (bool): Whether to apply gain to the tracks
            use_track_eq (bool): Whether to apply eq to the tracks
            use_track_compressor (bool): Whether to apply compressor to the tracks
            use_track_panner (bool): Whether to apply panner to the tracks
            use_fx_bus (bool): Whether to apply fx bus to the tracks
            use_master_bus (bool): Whether to apply master bus to the tracks.
            use_output_fader (bool): Whether to apply gain to the tracks.

        Returns:
            mixed_tracks (torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            master_bus (torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)

        """
        bs, num_tracks, seq_len = tracks.shape

        # move all tracks to batch dim for parallel processing
        tracks = tracks.view(-1, 1, seq_len)

        if tracks.sum() == 0:
            print("tracks is 0")
            print(tracks)

        # apply effects in series but all tracks at once
        if use_track_input_fader:
            tracks = gain(
                tracks,
                self.sample_rate,
                **track_param_dict["input_fader"],
            )
        if use_track_eq:
            tracks = parametric_eq(
                tracks,
                self.sample_rate,
                **track_param_dict["parametric_eq"],
            )
            if tracks.sum() == 0:
                print("eq is 0")
                print(tracks)
        if use_track_compressor:
            tracks = compressor(
                tracks,
                self.sample_rate,
                **track_param_dict["compressor"],
                lookahead_samples=2048,
            )
            if tracks.sum() == 0:
                print("compressor is 0")
                print(tracks)

        # restore tracks to original shape
        tracks = tracks.view(bs, num_tracks, seq_len)

        # restore tracks to original shape
        tracks = tracks.view(bs, num_tracks, seq_len)

        if use_track_panner:
            tracks = stereo_panner(
                tracks,
                self.sample_rate,
                **track_param_dict["stereo_panner"],
            )
        else:
            tracks = tracks.unsqueeze(1).repeat(1, 2, 1)

        # create stereo bus via summing
        master_bus = tracks.sum(dim=2)  # bs, 2, seq_len

        # apply stereo reveberation on an fx bus
        if use_fx_bus:
            fx_bus = stereo_bus(tracks, self.sample_rate, **track_param_dict["fx_bus"])
            fx_bus = noise_shaped_reverberation(
                fx_bus,
                self.sample_rate,
                **fx_bus_param_dict["reverberation"],
                num_samples=65536,
                num_bandpass_taps=1023,
            )
            master_bus += fx_bus

        if use_master_bus:
            # process Left channel
            master_bus = gain(
                master_bus,
                self.sample_rate,
                **master_bus_param_dict["input_fader"],
            )
            master_bus = parametric_eq(
                master_bus,
                self.sample_rate,
                **master_bus_param_dict["parametric_eq"],
            )

            # apply compressor to both channels
            master_bus = compressor(
                master_bus,
                self.sample_rate,
                **master_bus_param_dict["compressor"],
                lookahead_samples=1024,
            )

        if use_output_fader:
            master_bus = gain(
                master_bus,
                self.sample_rate,
                **master_bus_param_dict["output_fader"],
            )

        return tracks, master_bus

    def forward(
        self,
        tracks: torch.torch.Tensor,
        track_params: torch.torch.Tensor,
        fx_bus_params: torch.torch.Tensor,
        master_bus_params: torch.torch.Tensor,
        use_track_input_fader: bool = True,
        use_track_eq: bool = True,
        use_track_compressor: bool = True,
        use_track_panner: bool = True,
        use_master_bus: bool = True,
        use_fx_bus: bool = True,
        use_output_fader: bool = True,
    ):
        """Create a mix given a set of tracks and corresponding mixing parameters (0,1)

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            track_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_tracks, num_track_control_params)
            fx_bus_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_fx_bus_control_params)
            master_bus_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_master_bus_control_params)
            use_track_input_fader (bool): Whether to apply gain to the tracks
            use_track_eq (bool): Whether to apply eq to the tracks
            use_track_compressor (bool): Whether to apply compressor to the tracks
            use_track_panner (bool): Whether to apply panner to the tracks
            use_fx_bus (bool): Whether to apply fx bus to the tracks
            use_master_bus (bool): Whether to apply master bus to the tracks.
            use_output_fader (bool): Whether to apply gain to the tracks.

        Returns:
            mixed_tracks (torch.torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
            track_param_dict (dict): Denormalized track parameter values.
            fx_bus_param_dict (dict): Denormalized fx bus parameter values.
            master_bus_param_dict (dict): Denormalized master bus parameter values.
        """
        # extract and denormalize the parameters
        track_param_dict = {
            "input_fader": {
                "gain_db": track_params[..., 0],
            },
            "parametric_eq": {
                "low_shelf_gain_db": track_params[..., 1],
                "low_shelf_cutoff_freq": track_params[..., 2],
                "low_shelf_q_factor": track_params[..., 3],
                "band0_gain_db": track_params[..., 4],
                "band0_cutoff_freq": track_params[..., 5],
                "band0_q_factor": track_params[..., 6],
                "band1_gain_db": track_params[..., 7],
                "band1_cutoff_freq": track_params[..., 8],
                "band1_q_factor": track_params[..., 9],
                "band2_gain_db": track_params[..., 10],
                "band2_cutoff_freq": track_params[..., 11],
                "band2_q_factor": track_params[..., 12],
                "band3_gain_db": track_params[..., 13],
                "band3_cutoff_freq": track_params[..., 14],
                "band3_q_factor": track_params[..., 15],
                "high_shelf_gain_db": track_params[..., 16],
                "high_shelf_cutoff_freq": track_params[..., 17],
                "high_shelf_q_factor": track_params[..., 18],
            },
            # release and attack time must be the same
            "compressor": {
                "threshold_db": track_params[..., 19],
                "ratio": track_params[..., 20],
                "attack_ms": track_params[..., 21],
                "release_ms": track_params[..., 22],
                "knee_db": track_params[..., 23],
                "makeup_gain_db": track_params[..., 24],
            },
            "stereo_panner": {
                "pan": track_params[..., 25],
            },
            "fx_bus": {
                "send_db": track_params[..., 26],
            },
        }

        fx_bus_param_dict = {
            "reverberation": {
                "band0_gain": fx_bus_params[..., 0],
                "band1_gain": fx_bus_params[..., 1],
                "band2_gain": fx_bus_params[..., 2],
                "band3_gain": fx_bus_params[..., 3],
                "band4_gain": fx_bus_params[..., 4],
                "band5_gain": fx_bus_params[..., 5],
                "band6_gain": fx_bus_params[..., 6],
                "band7_gain": fx_bus_params[..., 7],
                "band8_gain": fx_bus_params[..., 8],
                "band9_gain": fx_bus_params[..., 9],
                "band10_gain": fx_bus_params[..., 10],
                "band11_gain": fx_bus_params[..., 11],
                "band0_decay": fx_bus_params[..., 12],
                "band1_decay": fx_bus_params[..., 13],
                "band2_decay": fx_bus_params[..., 14],
                "band3_decay": fx_bus_params[..., 15],
                "band4_decay": fx_bus_params[..., 16],
                "band5_decay": fx_bus_params[..., 17],
                "band6_decay": fx_bus_params[..., 18],
                "band7_decay": fx_bus_params[..., 19],
                "band8_decay": fx_bus_params[..., 20],
                "band9_decay": fx_bus_params[..., 21],
                "band10_decay": fx_bus_params[..., 22],
                "band11_decay": fx_bus_params[..., 23],
                "mix": torch.ones_like(fx_bus_params[..., 24]),
            },
        }

        master_bus_param_dict = {
            "parametric_eq": {
                "low_shelf_gain_db": master_bus_params[..., 0],
                "low_shelf_cutoff_freq": master_bus_params[..., 1],
                "low_shelf_q_factor": master_bus_params[..., 2],
                "band0_gain_db": master_bus_params[..., 3],
                "band0_cutoff_freq": master_bus_params[..., 4],
                "band0_q_factor": master_bus_params[..., 5],
                "band1_gain_db": master_bus_params[..., 6],
                "band1_cutoff_freq": master_bus_params[..., 7],
                "band1_q_factor": master_bus_params[..., 8],
                "band2_gain_db": master_bus_params[..., 9],
                "band2_cutoff_freq": master_bus_params[..., 10],
                "band2_q_factor": master_bus_params[..., 11],
                "band3_gain_db": master_bus_params[..., 12],
                "band3_cutoff_freq": master_bus_params[..., 13],
                "band3_q_factor": master_bus_params[..., 14],
                "high_shelf_gain_db": master_bus_params[..., 15],
                "high_shelf_cutoff_freq": master_bus_params[..., 16],
                "high_shelf_q_factor": master_bus_params[..., 17],
            },
            # release and attack time must be the same
            "compressor": {
                "threshold_db": master_bus_params[..., 18],
                "ratio": master_bus_params[..., 19],
                "attack_ms": master_bus_params[..., 20],
                "release_ms": master_bus_params[..., 21],
                "knee_db": master_bus_params[..., 22],
                "makeup_gain_db": master_bus_params[..., 23],
            },
            "output_fader": {
                "gain_db": master_bus_params[..., 24],
            },
            "input_fader": {
                "gain_db": master_bus_params[..., 25],
            },
        }

        track_param_dict = denormalize_parameters(track_param_dict, self.param_ranges)
        fx_bus_param_dict = denormalize_parameters(fx_bus_param_dict, self.param_ranges)
        master_bus_param_dict = denormalize_parameters(
            master_bus_param_dict, self.param_ranges
        )

        mixed_tracks, mix = self.forward_mix_console(
            tracks,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
            use_track_input_fader=use_track_input_fader,
            use_track_eq=use_track_eq,
            use_track_compressor=use_track_compressor,
            use_track_panner=use_track_panner,
            use_fx_bus=use_fx_bus,
            use_master_bus=use_master_bus,
            use_output_fader=use_output_fader,
        )
        return (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        )


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

    def forward(self, x: torch.Tensor, mix_console: torch.nn.Module):
        """Take a tensor of mixes, separate, and then remix.

        Args:
            x (torch.Tensor): Tensor of mixes with shape (batch, 2, samples)
            mix_console (torch.nn.Module): MixConsole module

        Returns:
            remix (torch.Tensor): Tensor of remixes with shape (batch, 2, samples)
            track_params (torch.Tensor): Tensor of track params with shape (batch, 8, num_track_control_params)
            fx_bus_params (torch.Tensor): Tensor of fx bus params with shape (batch, num_fx_bus_control_params)
            master_bus_params (torch.Tensor): Tensor of master bus params with shape (batch, num_master_bus_control_params)
        """
        bs, chs, seq_len = x.size()

        # separate
        with torch.no_grad():
            sources = self.stem_separator(x)  # bs, 4, 2, seq_len
        sum_mix = sources.sum(dim=1)  # bs, 2, seq_len

        # convert sources to mono tracks
        tracks = sources.view(bs, 8, -1)

        # provide some headroom before mixing
        tracks *= 10 ** (-48.0 / 20.0)

        # generate random mix parameters
        track_params = torch.rand(bs, 8, mix_console.num_track_control_params).type_as(
            x
        )
        fx_bus_params = torch.rand(bs, mix_console.num_fx_bus_control_params).type_as(x)
        master_bus_params = torch.rand(
            bs, mix_console.num_master_bus_control_params
        ).type_as(x)

        # the forward expects params in range of (0,1)
        with torch.no_grad():
            result = mix_console(
                tracks,
                track_params,
                fx_bus_params,
                master_bus_params,
                use_output_fader=False,
            )

        # get the remix
        remix = result[1]

        # clip via tanh if above 4.0
        remix = torch.tanh((1 / 4.0) * remix)
        remix *= 4.0

        return remix, track_params, fx_bus_params, master_bus_params


class ParameterProjector(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_tracks: int,
        num_track_control_params: int,
        num_fx_bus_control_params: int,
        num_master_bus_control_params: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tracks = num_tracks

        self.track_projector = torch.nn.Linear(
            embed_dim,
            num_tracks * num_track_control_params,
        )
        self.fx_bus_projector = torch.nn.Linear(
            embed_dim,
            num_fx_bus_control_params,
        )
        self.master_bus_projector = torch.nn.Linear(
            embed_dim,
            num_master_bus_control_params,
        )

    def forward(self, z: torch.Tensor):
        bs, embed_dim = z.size()

        track_params = torch.sigmoid(self.track_projector(z))
        track_params = track_params.view(bs, self.num_tracks, -1)
        fx_bus_params = torch.sigmoid(self.fx_bus_projector(z))
        master_bus_params = torch.sigmoid(self.master_bus_projector(z))

        return track_params, fx_bus_params, master_bus_params


class WaveformEncoder(torch.nn.Module):

    def __init__(
        self,
        n_inputs=1,
        embed_dim: int = 1024,
        encoder_batchnorm: bool = True,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.embed_dim = embed_dim
        self.encoder_batchnorm = encoder_batchnorm
        self.model = TCN(n_inputs, embed_dim)

    def forward(self, x: torch.Tensor):
        return self.model(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class WaveformTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        n_inputs: int = 1,
        block_size: int = 1024,
        embed_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.cls = torch.nn.Parameter(torch.randn(1, 1, block_size))
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=block_size,
            nhead=nhead,
            batch_first=True,
        )
        self.model = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor):
        bs, chs, seq_len = x.size()
        # chunk the input waveform into non-overlapping blocks
        x = x.unfold(-1, self.block_size, self.block_size)

        # move channels to sequence dim
        x = x.view(bs, chs * x.shape[-2], x.shape[-1])

        # add cls token
        cls_token = self.cls.repeat(bs, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        z = self.model(x)

        return z[:, 0, :]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class WaveformTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        n_inputs: int = 1,
        block_size: int = 1024,
        embed_dim: int = 512,
        nhead: int = 8,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.cls = torch.nn.Parameter(torch.randn(1, 1, block_size))
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=block_size,
            nhead=nhead,
            batch_first=True,
        )
        self.model = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(self, x: torch.Tensor):
        bs, chs, seq_len = x.size()
        # chunk the input waveform into non-overlapping blocks
        x = x.unfold(-1, self.block_size, self.block_size)

        # move channels to sequence dim
        x = x.view(bs, chs * x.shape[-2], x.shape[-1])

        # add cls token
        cls_token = self.cls.repeat(bs, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        z = self.model(x)

        return z[:, 0, :]


class SpectrogramEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        n_inputs: int = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        input_batchnorm: bool = False,
        encoder_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_inputs = n_inputs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_batchnorm = input_batchnorm
        window_length = int(n_fft)
        self.register_buffer("window", torch.hann_window(window_length=window_length))

        # self.model = torchvision.models.resnet.resnet18(num_classes=embed_dim)
        self.model = Cnn14(
            n_inputs=n_inputs,
            num_classes=embed_dim,
            use_batchnorm=encoder_batchnorm,
        )

        if self.input_batchnorm:
            self.bn = torch.nn.BatchNorm2d(3)
        else:
            self.bn = torch.nn.Identity()

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        """Process waveform as a spectrogram and return single embedding.

        Args:
            x (torch.torch.Tensor): Monophonic waveform torch.Tensor of shape (bs, chs, seq_len).

        Returns:
            embed (torch.Tenesor): Embedding torch.Tensor of shape (bs, embed_dim)
        """

        bs, chs, seq_len = x.size()

        # move channels to batch dim
        x = x.view(-1, seq_len)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        X = X.view(bs, chs, X.shape[-2], X.shape[-1])

        X = torch.pow(X.abs() + 1e-8, 0.3)
        # X = X.repeat(1, 3, 1, 1)  # add dummy channels (3)

        # apply normalization
        if self.input_batchnorm:
            X = self.bn(X)

        # process with CNN
        embeds = self.model(X)
        # print(embeds.shape)
        return embeds


class TransformerController(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_track_control_params: int,
        num_fx_bus_control_params: int,
        num_master_bus_control_params: int,
        num_layers: int = 6,
        nhead: int = 8,
        use_fx_bus: bool = False,
        use_master_bus: bool = False,
    ) -> None:
        """Transformer based Controller that predicts mix parameters given track and reference mix embeddings.

        Args:
            embed_dim (int): Embedding dim for tracks and mix.
            num_control_params (int): Number of control parameters for each track.
            num_layers (int): Number of Transformer layers.
            nhead (int): Number of attention heads in each layer.
            use_fx_bus (bool): Whether to use the FX bus.
            use_master_bus (bool): Whether to use the master bus.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_track_control_params = num_track_control_params
        self.num_fx_bus_control_params = num_fx_bus_control_params
        self.num_master_bus_control_params = num_master_bus_control_params
        self.num_layers = num_layers
        self.nhead = nhead
        self.use_fx_bus = use_fx_bus
        self.use_master_bus = use_master_bus

        self.track_embedding = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mix_embedding = torch.nn.Parameter(torch.randn(1, 2, embed_dim))
        self.fx_bus_embedding = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.master_bus_embedding = torch.nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, dropout=0.0
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.track_projection = torch.nn.Linear(embed_dim, num_track_control_params)
        self.fx_bus_projection = torch.nn.Linear(embed_dim, num_fx_bus_control_params)
        self.master_bus_projection = torch.nn.Linear(
            embed_dim, num_master_bus_control_params
        )

    def forward(
        self,
        track_embeds: torch.torch.Tensor,
        mix_embeds: torch.torch.Tensor,
        track_padding_mask: Optional[torch.Tensor] = None,
    ):
        """Predict mix parameters given track and reference mix embeddings.

        Args:
            track_embeds (torch.torch.Tensor): Embeddings for each track with shape (bs, num_tracks, embed_dim)
            mix_embeds (torch.torch.Tensor): Embeddings for the reference mix with shape (bs, 2, embed_dim)
            track_padding_mask (Optional[torch.Tensor]): Mask for the track embeddings with shape (bs, num_tracks)

        Returns:
            pred_track_params (torch.torch.Tensor): Predicted track parameters with shape (bs, num_tracks, num_control_params)
            pred_fx_bus_params (torch.torch.Tensor): Predicted fx bus parameters with shape (bs, num_fx_bus_control_params)
            pred_master_bus_params (torch.torch.Tensor): Predicted master bus parameters with shape (bs, num_master_bus_control_params)
        """
        bs, num_tracks, embed_dim = track_embeds.size()

        # apply learned embeddings to both input embeddings
        track_embeds += self.track_embedding.repeat(bs, num_tracks, 1)
        mix_embeds += self.mix_embedding.repeat(bs, 1, 1)

        # concat embeds into single "sequence"
        embeds = torch.cat((track_embeds, mix_embeds), dim=1)  # bs, seq_len, embed_dim
        embeds = torch.cat((embeds, self.fx_bus_embedding.repeat(bs, 1, 1)), dim=1)
        embeds = torch.cat((embeds, self.master_bus_embedding.repeat(bs, 1, 1)), dim=1)

        # add to padding mask for mix_embeds, fx and master bus so they are attended to
        if track_padding_mask is not None:
            track_padding_mask = torch.cat(
                (
                    track_padding_mask,
                    torch.zeros((bs, 4), dtype=torch.bool).type_as(track_padding_mask),
                ),
                dim=1,
            )

        # generate output embeds with transformer, project and bound 0 - 1
        pred_params = self.transformer_encoder(
            embeds, src_key_padding_mask=track_padding_mask
        )
        pred_track_params = torch.sigmoid(
            self.track_projection(pred_params[:, :num_tracks, :])
        )
        # print(pred_track_params)
        pred_fx_bus_params = torch.sigmoid(
            self.fx_bus_projection(pred_params[:, -2, :])
        )
        pred_master_bus_params = torch.sigmoid(
            self.master_bus_projection(pred_params[:, -1, :])
        )

        return pred_track_params, pred_fx_bus_params, pred_master_bus_params
