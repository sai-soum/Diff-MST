import torch
import torchvision
from dasp_pytorch.functional import gain, stereo_panner, compressor, parametric_eq


class TransformerController(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_control_params: int,
        num_layers: int = 6,
        nhead: int = 8,
    ) -> None:
        """Transformer based Controller that predicts mix parameters given track and reference mix embeddings.

        Args:
            embed_dim (int): Embedding dim for tracks and mix.
            num_control_params (int): Number of control parameters for each track.
            num_layers (int): Number of Transformer layers.
            nhead (int): Number of attention heads in each layer.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.track_embedding = torch.nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mix_embedding = torch.nn.Parameter(torch.randn(1, 2, embed_dim))

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.projection = torch.nn.Linear(embed_dim, num_control_params)

    def forward(self, track_embeds: torch.Tensor, mix_embeds: torch.Tensor):
        bs, num_tracks, embed_dim = track_embeds.size()

        # apply learned embeddings to both input embeddings
        track_embeds += self.track_embedding.repeat(bs, num_tracks, 1)
        mix_embeds += self.mix_embedding.repeat(bs, 1, 1)

        # concat embeds into single "sequence"
        embeds = torch.cat((track_embeds, mix_embeds), dim=1)  # bs, seq_len, embed_dim

        # generate output embeds with transformer, project and bound 0 - 1
        pred_params = torch.sigmoid(self.projection(self.transformer_encoder(embeds)))
        pred_params = pred_params[:, :num_tracks, :]  # ignore the outputs of mix embeds

        return pred_params


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process waveform as a spectrogram and return single embedding.

        Args:
            x (torch.Tensor): Monophonic waveform tensor of shape (bs, seq_len).

        Returns:
            embed (torch.Tenesor): Embedding tensor of shape (bs, embed_dim)
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


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def denormalize_parameters(param_dict: dict, param_ranges: dict):
    """Given parameters on (0,1) restore them to the ranges expected by the denoiser."""
    denorm_param_dict = {}
    for effect_name, effect_param_dict in param_dict.items():
        denorm_param_dict[effect_name] = {}
        for param_name, param_val in effect_param_dict.items():
            param_val_denorm = denormalize(
                param_val,
                param_ranges[effect_name][param_name][1],
                param_ranges[effect_name][param_name][0],
            )
            denorm_param_dict[effect_name][param_name] = param_val_denorm
    return denorm_param_dict


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
        """Expects the param_dict has denormalized parameters."""
        # apply effects in series and all tracks in parallel
        tracks = gain(tracks, **param_dict["input_gain"])
        tracks = stereo_panner(tracks, **param_dict["stereo_panner"])
        return tracks.sum(dim=1)

    def forward(self, tracks: torch.Tensor, mix_params: torch.Tensor):
        """Create a mix given a set of tracks and corresponding mixing parameters (0,1)

        Args:
            tracks (torch.Tensor): Audio tracks with shape (bs, num_tracks, seq_len)
            mix_params (torch.Tensor): Parameter tensor with shape (bs, num_tracks, num_control_params)

        Returns:
            mix (torch.Tensor): Final stereo mix of the input tracks with shape (bs, 2, seq_len)
            param_dict (dict): Denormalized parameter values.
        """
        # extract and denormalize the parameters
        param_dict = {
            "input_gain": {
                "gain_db": mix_params[:,:, 0],  # bs, num_tracks, 1
            },
            "stereo_panner": {
                "pan": mix_params[:,:, 1],
            },
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        mix = self.forward_mix_console(tracks, param_dict)
        return mix, param_dict


class AdvancedMixConsole(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_control_params = 26

    def forward_mix_console(self, tracks: torch.Tensor, param_dict: dict):
        # apply effects in series but all tracks at once
        tracks = gain(tracks, **param_dict["input_gain"])
        tracks = parametric_eq(tracks, **param_dict["parametric_eq"])
        tracks = compressor(tracks, **param_dict["compressor"])
        tracks = stereo_panner(tracks, **param_dict["stereo_panner"])
        return tracks.sum(dim=1)

    def forward(self, tracks: torch.Tensor, mix_params: torch.Tensor):
        # extract and denormalize the parameters
        param_dict = {
            "input_gain": {
                "gain_db": mix_params[..., 0],
            },
            "parametric_eq": {
                "low_shelf_gain_db": mix_params[..., 1],
                "low_shelf_cutoff_freq": mix_params[..., 2],
                "low_shelf_q_factor": mix_params[..., 3],
                "first_band_gain_db": mix_params[..., 4],
                "first_band_cutoff_freq": mix_params[..., 5],
                "first_band_q_factor": mix_params[..., 6],
                "second_band_gain_db": mix_params[..., 7],
                "second_band_cutoff_freq": mix_params[..., 8],
                "second_band_q_factor": mix_params[..., 9],
                "third_band_gain_db": mix_params[..., 10],
                "third_band_cutoff_freq": mix_params[..., 11],
                "third_band_q_factor": mix_params[..., 12],
                "fourth_band_gain_db": mix_params[..., 13],
                "fourth_band_cutoff_freq": mix_params[..., 14],
                "fourth_band_q_factor": mix_params[..., 15],
                "high_shelf_gain_db": mix_params[..., 16],
                "high_shelf_cutoff_freq": mix_params[..., 17],
                "high_shelf_q_factor": mix_params[..., 18],
            },
            "compressor": {
                "threshold_db": mix_params[..., 19],
                "ratio": mix_params[..., 20],
                "attack_ms": mix_params[..., 21],
                "release_ms": mix_params[..., 22],
                "knee_db": mix_params[..., 23],
                "makeup_gain_db": mix_params[..., 24],
            },
            "stereo_panner": {
                "pan": mix_params[..., 25],
            },
        }
        param_dict = denormalize_parameters(param_dict, self.param_ranges)
        mix = self.forward_mix_console(tracks, param_dict)
        return mix


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

    def forward(self, tracks: torch.Tensor, ref_mix: torch.Tensor):
        bs, num_tracks, seq_len = tracks.size()

        # first process the tracks
        track_embeds = self.track_encoder(tracks.view(bs * num_tracks, -1))
        track_embeds = track_embeds.view(bs, num_tracks, -1)  # restore

        # process the reference mix
        mix_embeds = self.mix_encoder(ref_mix.view(bs * 2, -1))
        mix_embeds = mix_embeds.view(bs, 2, -1)  # restore

        # controller will predict mix parameters for each stem based on embeds
        mix_params = self.controller(track_embeds, mix_embeds)

        # create a mix using the predicted parameters
        pred_mix, mix_params = self.mix_console(tracks, mix_params)

        return pred_mix, mix_params
