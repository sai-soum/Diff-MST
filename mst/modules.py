import torch
import torchvision
from typing import Callable, Optional, List
import torchaudio
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
        mix_console: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.track_encoder = track_encoder
        self.mix_encoder = mix_encoder
        self.controller = controller
        self.mix_console = mix_console

    def forward(self, tracks: torch.torch.Tensor, ref_mix: torch.torch.Tensor):
        bs, num_tracks, seq_len = tracks.size()

        # first process the tracks
        track_embeds = self.track_encoder(tracks.view(bs * num_tracks, -1))
        track_embeds = track_embeds.view(bs, num_tracks, -1)  # restore

        # process the reference mix
        mix_embeds = self.mix_encoder(ref_mix.view(bs * 2, -1))
        mix_embeds = mix_embeds.view(bs, 2, -1)  # restore

        # controller will predict mix parameters for each stem based on embeds
        track_params, fx_bus_params, master_bus_params = self.controller(
            track_embeds, mix_embeds
        )

        # create a mix using the predicted parameters
        (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        ) = self.mix_console(tracks, track_params, fx_bus_params, master_bus_params)

        return (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        )


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


def denormalize_parameters(param_dict: dict, param_ranges: dict):
    """Given parameters on (0,1) restore them to the ranges expected by the denoiser."""
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


class TCNMixConsole(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        tcn: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.tcn = tcn
        self.sample_rate = sample_rate
        self.num_control_params = tcn.cond_dim

    def forward_mix_console(self, tracks: torch.Tensor, mix_params: torch.Tensor):
        """Apply the mix console to a set of tracks.
        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            param_dict (dict): Denormalized parameter values.
        Returns:
            mixed_tracks (torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
        """
        # apply effects in series and all tracks in parallel
        bs, num_tracks, seq_len = tracks.size()
        tracks = tracks.view(
            bs * num_tracks, 1, -1
        )  # move tracks and conditioning to the batch dim
        mix_params = mix_params.reshape(bs * num_tracks, self.num_control_params)
        tracks = self.tcn(tracks, mix_params)  # process all tracks in parallel
        tracks = tracks.view(bs, num_tracks, 2, -1)  # move tracks back
        return tracks, tracks.sum(dim=1)

    def forward(self, tracks: torch.Tensor, mix_params: torch.torch.Tensor):
        """Create a mix given a set of tracks and corresponding mixing parameters (0,1)

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            mix_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_tracks, num_control_params)

        Returns:
            mixed_tracks (torch.torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
            param_dict (dict): Denormalized parameter values.
        """
        param_dict = {}
        for n in range(self.num_control_params):
            param_dict[f"param_{n}"] = mix_params[:, :, n]
        mixed_tracks, mix = self.forward_mix_console(tracks, mix_params)
        return mixed_tracks, mix, param_dict


class BasicMixConsole(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -80.0,
        max_gain_db: float = 24.0,
        min_pan: float = 0.0,
        max_pan: float = 1.0,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.param_ranges = {
            "input_gain": {"gain_db": (min_gain_db, max_gain_db)},
            "stereo_panner": {"pan": (min_pan, max_pan)},
        }
        self.num_control_params = 2

    def forward_mix_console(self, tracks: torch.Tensor, param_dict: dict):
        """Apply the mix console to a set of tracks.
        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            param_dict (dict): Denormalized parameter values.
        Returns:
            mixed_tracks (torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
        """
        # apply effects in series and all tracks in parallel
        bs, chs, seq_len = tracks.size()
        tracks = gain(tracks, **param_dict["input_gain"])
        tracks = stereo_panner(tracks, **param_dict["stereo_panner"])
        return tracks, tracks.sum(dim=2)

    def forward(self, tracks: torch.Tensor, mix_params: torch.torch.Tensor):
        """Create a mix given a set of tracks and corresponding mixing parameters (0,1)

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            mix_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_tracks, num_control_params)

        Returns:
            mixed_tracks (torch.torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
            param_dict (dict): Denormalized parameter values.
        """
        # extract and denormalize the parameters
        param_dict = {
            "input_gain": {
                "gain_db": mix_params[:, :, 0],  # bs, num_tracks, 1
            },
            "stereo_panner": {
                "pan": mix_params[:, :, 1],
            },
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        mixed_tracks, mix = self.forward_mix_console(tracks, param_dict)
        return mixed_tracks, mix, param_dict


class AdvancedMixConsole(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
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
            "input_gain": {"gain_db": (min_gain_db, max_gain_db)},
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
                "attack_ms": (1.0, 1000.0),
                "release_ms": (1.0, 1000.0),
                "knee_db": (3.0, 24.0),
                "makeup_gain_db": (0.0, 24.0),
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
        self.num_master_bus_control_params = 24

    def forward_mix_console(
        self,
        tracks: torch.torch.Tensor,
        track_param_dict: dict,
        fx_bus_param_dict: dict,
        master_bus_param_dict: dict,
        use_track_gain: bool = True,
        use_track_eq: bool = True,
        use_track_compressor: bool = True,
        use_track_panner: bool = True,
        use_fx_bus: bool = True,
        use_master_bus: bool = True,
    ):
        """

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            param_dict (dict): Denormalized parameter values.
        Returns:
            mixed_tracks (torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
        """
        bs, num_tracks, seq_len = tracks.shape
        # apply effects in series but all tracks at once
        if use_track_gain:
            tracks = gain(tracks, **track_param_dict["input_gain"])
        if use_track_eq:
            tracks = parametric_eq(
                tracks, self.sample_rate, **track_param_dict["parametric_eq"]
            )
        if use_track_compressor:
            tracks = compressor(
                tracks, self.sample_rate, **track_param_dict["compressor"]
            )

        if use_track_panner:
            tracks = stereo_panner(tracks, **track_param_dict["stereo_panner"])

        # create stereo bus via summing
        master_bus = tracks.sum(dim=2)

        # apply stereo reveberation on an fx bus
        if use_fx_bus:
            fx_bus = stereo_bus(tracks, **track_param_dict["fx_bus"])
            fx_bus = noise_shaped_reverberation(
                fx_bus, self.sample_rate, **fx_bus_param_dict["reverberation"]
            )
            master_bus += fx_bus

        # split master into left and right for linked processing (same parameters)
        master_bus_L = master_bus[:, 0:1]
        master_bus_R = master_bus[:, 1:2]

        # process Left channel
        if use_master_bus:
            master_bus_L = parametric_eq(
                master_bus_L, self.sample_rate, **master_bus_param_dict["parametric_eq"]
            )
            master_bus_L = compressor(
                master_bus_L, self.sample_rate, **master_bus_param_dict["compressor"]
            )
            # process Right channel
            master_bus_R = parametric_eq(
                master_bus_R, self.sample_rate, **master_bus_param_dict["parametric_eq"]
            )
            master_bus_R = compressor(
                master_bus_R, self.sample_rate, **master_bus_param_dict["compressor"]
            )

        # recompose stereo mix
        master_bus = torch.cat((master_bus_L, master_bus_R), dim=1)

        return tracks, master_bus

    def forward(
        self,
        tracks: torch.torch.Tensor,
        track_params: torch.torch.Tensor,
        fx_bus_params: torch.torch.Tensor,
        master_bus_params: torch.torch.Tensor,
        use_track_gain: bool = True,
        use_track_eq: bool = True,
        use_track_compressor: bool = True,
        use_master_bus: bool = True,
        use_fx_bus: bool = True,
    ):
        """Create a mix given a set of tracks and corresponding mixing parameters (0,1)

        Args:
            tracks (torch.torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            mix_params (torch.torch.Tensor): Parameter torch.Tensor with shape (bs, num_tracks, num_control_params)

        Returns:
            mixed_tracks (torch.torch.Tensor): Mixed tracks with shape (bs, num_tracks, seq_len)
            mix (torch.torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
            param_dict (dict): Denormalized parameter values.
        """
        # extract and denormalize the parameters
        track_param_dict = {
            "input_gain": {
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
                "mix": fx_bus_params[..., 24],
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
            use_track_gain,
            use_track_eq,
            use_track_compressor,
            use_fx_bus,
            use_master_bus,
        )
        return (
            mixed_tracks,
            mix,
            track_param_dict,
            fx_bus_param_dict,
            master_bus_param_dict,
        )


class SpectrogramResNetEncoder(torch.nn.Module):
    def __init__(
        self,
        embed_dim=128,
        n_fft: int = 2048,
        hop_length: int = 512,
        input_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.input_batchnorm = input_batchnorm
        window_length = int(n_fft)
        self.register_buffer("window", torch.hann_window(window_length=window_length))
        self.model = torchvision.models.resnet.resnet18(num_classes=embed_dim)
        if self.input_batchnorm:
            self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x: torch.torch.Tensor) -> torch.torch.Tensor:
        """Process waveform as a spectrogram and return single embedding.

        Args:
            x (torch.torch.Tensor): Monophonic waveform torch.Tensor of shape (bs, seq_len).

        Returns:
            embed (torch.Tenesor): Embedding torch.Tensor of shape (bs, embed_dim)
        """
        bs, seq_len = x.size()

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        X = X.view(bs, 1, X.shape[-2], X.shape[-1])
        X = torch.pow(X.abs() + 1e-8, 0.3)
        X = X.repeat(1, 3, 1, 1)  # add dummy channels (3)

        # apply normalization
        if self.input_batchnorm:
            X = self.bn(X)

        # process with CNN
        embeds = self.model(X)
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
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
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

    def forward(self, track_embeds: torch.torch.Tensor, mix_embeds: torch.torch.Tensor):
        bs, num_tracks, embed_dim = track_embeds.size()

        # apply learned embeddings to both input embeddings
        track_embeds += self.track_embedding.repeat(bs, num_tracks, 1)
        mix_embeds += self.mix_embedding.repeat(bs, 1, 1)

        # concat embeds into single "sequence"
        embeds = torch.cat((track_embeds, mix_embeds), dim=1)  # bs, seq_len, embed_dim
        embeds = torch.cat((embeds, self.fx_bus_embedding.repeat(bs, 1, 1)), dim=1)
        embeds = torch.cat((embeds, self.master_bus_embedding.repeat(bs, 1, 1)), dim=1)

        # generate output embeds with transformer, project and bound 0 - 1
        pred_params = self.transformer_encoder(embeds)
        pred_track_params = torch.sigmoid(
            self.track_projection(pred_params[:, :num_tracks, :])
        )
        pred_fx_bus_params = torch.sigmoid(
            self.fx_bus_projection(pred_params[:, -2, :])
        )
        pred_master_bus_params = torch.sigmoid(
            self.master_bus_projection(pred_params[:, -1, :])
        )

        return pred_track_params, pred_fx_bus_params, pred_master_bus_params


def center_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        start = (x.size(-1) - length) // 2
        stop = start + length
        x = x[..., start:stop]
    return x


def causal_crop(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.size(-1) != length:
        assert x.size(-1) > length
        stop = x.size(-1) - 1
        start = stop - length
        x = x[..., start:stop]
    return x


class FiLM(torch.nn.Module):
    def __init__(
        self,
        cond_dim: int,  # dim of conditioning input
        num_features: int,  # dim of the conv channel
        use_bn: bool = True,
    ) -> None:  # TODO(cm): check what this default value should be
        super().__init__()
        self.num_features = num_features
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, 2 * num_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g[:, :, None]
        b = b[:, :, None]
        if self.use_bn:
            x = self.bn(x)  # Apply batchnorm without affine
        x = (x * g) + b  # Then apply conditional affine
        return x


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        padding: Optional[int] = 0,
        use_ln: bool = False,
        temporal_dim: Optional[int] = None,
        use_act: bool = True,
        use_res: bool = True,
        cond_dim: int = 0,
        use_film_bn: bool = True,
        crop_fn: Callable[[torch.Tensor, int], torch.Tensor] = causal_crop,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.use_ln = use_ln
        self.temporal_dim = temporal_dim
        self.use_act = use_act
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.crop_fn = crop_fn

        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
        self.padding = padding

        self.ln = None
        if use_ln:
            assert temporal_dim is not None and temporal_dim > 0
            self.ln = torch.nn.LayerNorm(
                [in_ch, temporal_dim], elementwise_affine=False
            )

        self.act = None
        if use_act:
            self.act = torch.nn.PReLU(out_ch)

        self.conv = torch.nn.Conv1d(
            in_ch,
            out_ch,
            (kernel_size,),
            stride=(stride,),
            padding=padding,
            dilation=(dilation,),
            bias=True,
        )
        self.res = None
        if use_res:
            self.res = torch.nn.Conv1d(
                in_ch, out_ch, kernel_size=(1,), stride=(stride,), bias=False
            )

        self.film = None
        if cond_dim > 0:
            self.film = FiLM(cond_dim, out_ch, use_bn=use_film_bn)

    def is_conditional(self) -> bool:
        return self.cond_dim > 0

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x_in = x
        if self.ln is not None:
            assert x.size(1) == self.in_ch
            assert x.size(2) == self.temporal_dim
            x = self.ln(x)
        x = self.conv(x)
        if self.is_conditional():
            assert cond is not None
            x = self.film(x, cond)
        if self.act is not None:
            x = self.act(x)
        if self.res is not None:
            res = self.res(x_in)
            x_res = self.crop_fn(res, x.size(-1))
            x += x_res
        return x


class TCN(torch.nn.Module):
    def __init__(
        self,
        out_channels: List[int],
        dilations: Optional[List[int]] = None,
        in_ch: int = 1,
        kernel_size: int = 13,
        strides: Optional[List[int]] = None,
        padding: Optional[int] = 0,
        use_ln: bool = False,
        temporal_dims: Optional[List[int]] = None,
        use_act: bool = True,
        use_res: bool = True,
        cond_dim: int = 0,
        use_film_bn: bool = False,
        crop_fn: Callable[[torch.Tensor, int], torch.Tensor] = causal_crop,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.in_ch = in_ch
        self.out_ch = out_channels[-1]
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_ln = use_ln
        self.temporal_dims = temporal_dims  # TODO(cm): calculate automatically
        self.use_act = use_act
        self.use_res = use_res
        self.cond_dim = cond_dim
        self.use_film_bn = use_film_bn
        self.crop_fn = crop_fn
        # TODO(cm): padding warning

        self.n_blocks = len(out_channels)
        if dilations is None:
            dilations = [4**idx for idx in range(self.n_blocks)]
        assert len(dilations) == self.n_blocks
        self.dilations = dilations

        if strides is None:
            strides = [1] * self.n_blocks
        assert len(strides) == self.n_blocks
        self.strides = strides

        if use_ln:
            assert temporal_dims is not None
            assert len(temporal_dims) == self.n_blocks

        self.blocks = torch.nn.ModuleList()
        block_out_ch = None
        for idx, (curr_out_ch, dil, stride) in enumerate(
            zip(out_channels, dilations, strides)
        ):
            if idx == 0:
                block_in_ch = in_ch
            else:
                block_in_ch = block_out_ch
            block_out_ch = curr_out_ch

            temp_dim = None
            if temporal_dims is not None:
                temp_dim = temporal_dims[idx]

            self.blocks.append(
                TCNBlock(
                    block_in_ch,
                    block_out_ch,
                    kernel_size,
                    dil,
                    stride,
                    padding,
                    use_ln,
                    temp_dim,
                    use_act,
                    use_res,
                    cond_dim,
                    use_film_bn,
                    crop_fn,
                )
            )

    def is_conditional(self) -> bool:
        return self.cond_dim > 0

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        if self.is_conditional():
            assert cond is not None
            assert cond.shape == (x.size(0), self.cond_dim)  # (batch_size, cond_dim)
        for block in self.blocks:
            x = block(x, cond)
        return x

    def calc_receptive_field(self) -> int:
        """Compute the receptive field in samples."""
        assert all(_ == 1 for _ in self.strides)  # TODO(cm): add support for dsTCN
        assert self.dilations[0] == 1  # TODO(cm): add support for >1 starting dilation
        rf = self.kernel_size
        for dil in self.dilations[1:]:
            rf = rf + ((self.kernel_size - 1) * dil)
        return rf
