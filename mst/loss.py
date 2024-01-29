import torch
import librosa
import laion_clap
from typing import Callable, List

from mst.filter import barkscale_fbanks

import yaml
from mst.fx_encoder import FXencoder

from mst.modules import SpectrogramEncoder
from mst.quality_system import QualityEstimationSystem


def compute_mid_side(x: torch.Tensor):
    x_mid = x[:, 0, :] + x[:, 1, :]
    x_side = x[:, 0, :] - x[:, 1, :]
    return x_mid, x_side


def compute_melspectrum(
    x: torch.Tensor,
    sample_rate: int = 44100,
    fft_size: int = 32768,
    n_bins: int = 128,
    **kwargs,
):
    """Compute mel-spectrogram.

    Args:
        x: (bs, 2, seq_len)
        sample_rate: sample rate of audio
        fft_size: size of fft
        n_bins: number of mel bins

    Returns:
        X: (bs, n_bins)

    """
    fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
    fb = torch.tensor(fb).unsqueeze(0).type_as(x)

    x = x.mean(dim=1, keepdim=True)
    X = torch.fft.rfft(x, n=fft_size, dim=-1)
    X = torch.abs(X)
    X = torch.mean(X, dim=1, keepdim=True)  # take mean over time
    X = X.permute(0, 2, 1)  # swap time and freq dims
    X = torch.matmul(fb, X)
    X = torch.log(X + 1e-8)

    return X


def compute_barkspectrum(
    x: torch.Tensor,
    fft_size: int = 32768,
    n_bands: int = 24,
    sample_rate: int = 44100,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    mode: str = "mid-side",
    **kwargs,
):
    """Compute bark-spectrogram.

    Args:
        x: (bs, 2, seq_len)
        fft_size: size of fft
        n_bands: number of bark bins
        sample_rate: sample rate of audio
        f_min: minimum frequency
        f_max: maximum frequency
        mode: "mono", "stereo", or "mid-side"

    Returns:
        X: (bs, 24)

    """
    # compute filterbank
    fb = barkscale_fbanks((fft_size // 2) + 1, f_min, f_max, n_bands, sample_rate)
    fb = fb.unsqueeze(0).type_as(x)
    fb = fb.permute(0, 2, 1)

    if mode == "mono":
        x = x.mean(dim=1)  # average over channels
        signals = [x]
    elif mode == "stereo":
        signals = [x[:, 0, :], x[:, 1, :]]
    elif mode == "mid-side":
        x_mid = x[:, 0, :] + x[:, 1, :]
        x_side = x[:, 0, :] - x[:, 1, :]
        signals = [x_mid, x_side]
    else:
        raise ValueError(f"Invalid mode {mode}")

    outputs = []
    for signal in signals:
        X = torch.stft(
            signal,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            return_complex=True,
        )  # compute stft
        X = torch.abs(X)  # take magnitude
        X = torch.mean(X, dim=-1, keepdim=True)  # take mean over time
        # X = X.permute(0, 2, 1)  # swap time and freq dims
        X = torch.matmul(fb, X)  # apply filterbank
        X = torch.log(X + 1e-8)
        # X = torch.cat([X, X_log], dim=-1)
        outputs.append(X)

    # stack into tensor
    X = torch.cat(outputs, dim=-1)

    return X


def compute_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, 1, seq_len)

    Returns:
        rms: (bs, )
    """
    rms = torch.sqrt(torch.mean(x**2, dim=-1).clamp(min=1e-8))
    return rms


def compute_crest_factor(x: torch.Tensor, **kwargs):
    """Compute crest factor as ratio of peak to rms energy in dB.

    Args:
        x: (bs, 2, seq_len)

    """
    num = torch.max(torch.abs(x), dim=-1)[0]
    den = compute_rms(x).clamp(min=1e-8)
    cf = 20 * torch.log10((num / den).clamp(min=1e-8))
    return cf


def compute_stereo_width(x: torch.Tensor, **kwargs):
    """Compute stereo width as ratio of energy in sum and difference signals.

    Args:
        x: (bs, 2, seq_len)

    """
    bs, chs, seq_len = x.size()

    assert chs == 2, "Input must be stereo"

    # compute sum and diff of stereo channels
    x_sum = x[:, 0, :] + x[:, 1, :]
    x_diff = x[:, 0, :] - x[:, 1, :]

    # compute power of sum and diff
    sum_energy = torch.mean(x_sum**2, dim=-1)
    diff_energy = torch.mean(x_diff**2, dim=-1)

    # compute stereo width as ratio
    stereo_width = diff_energy / sum_energy.clamp(min=1e-8)

    return stereo_width


def compute_stereo_imbalance(x: torch.Tensor, **kwargs):
    """Compute stereo imbalance as ratio of energy in left and right channels.

    Args:
        x: (bs, 2, seq_len)

    Returns:
        stereo_imbalance: (bs, )

    """
    left_energy = torch.mean(x[:, 0, :] ** 2, dim=-1)
    right_energy = torch.mean(x[:, 1, :] ** 2, dim=-1)

    stereo_imbalance = (right_energy - left_energy) / (
        right_energy + left_energy
    ).clamp(min=1e-8)

    return stereo_imbalance


class AudioFeatureLoss(torch.nn.Module):
    def __init__(
        self,
        weights: List[float],
        sample_rate: int,
        stem_separation: bool = False,
        use_clap: bool = False,
    ) -> None:
        """Compute loss using a set of differentiable audio features.

        Args:
            weights: weights for each feature
            sample_rate: sample rate of audio
            stem_separation: whether to compute loss on stems or mix

        Based on features proposed in:

        Man, B. D., et al.
        "An analysis and evaluation of audio features for multitrack music mixtures."
        (2014).

        """
        super().__init__()
        self.weights = weights
        self.sample_rate = sample_rate
        self.stem_separation = stem_separation
        self.sources_list = ["mix"]
        self.source_weights = [1.0]
        self.use_clap = use_clap

        self.transforms = [
            compute_rms,
            compute_crest_factor,
            compute_stereo_width,
            compute_stereo_imbalance,
            compute_barkspectrum,
        ]

        if self.use_clap:
            # instatiate pretrained CLAP model
            self.clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            self.clap_model.load_ckpt()  # download the default pretrained checkpoint.

            # freeze all parameters in model
            for param in self.clap_model.parameters():
                param.requires_grad = False

            def compute_clap_embeds(x: torch.Tensor, **kwargs):
                x = x.mean(dim=1)  # average over channels
                embed = self.clap_model.get_audio_embedding_from_data(
                    x=x, use_tensor=True
                )
                return embed

            self.transforms.append(compute_clap_embeds)

        assert len(self.transforms) == len(weights)

        if stem_separation:  # load pretrained stem separation model
            from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

            bundle = HDEMUCS_HIGH_MUSDB_PLUS
            self.stem_separator = bundle.get_model()
            self.stem_separator.train()

            # get sources list
            self.sources_list += list(self.stem_separator.sources)
            self.source_weights += [0.1, 0.1, 0.1, 0.1]

            # freeze all parameters in model
            # for param in self.stem_separator.parameters():
            #    param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        losses = {}

        # first separate stems if necessary
        if self.stem_separation:
            input_stems = self.stem_separator(input)
            target_stems = self.stem_separator(target)
            # bs, n_stems, chs, seq_len

            # add original mix to tensor of stems
            input_stems = torch.cat([input.unsqueeze(1), input_stems], dim=1)
            target_stems = torch.cat([target.unsqueeze(1), target_stems], dim=1)
        else:
            # reshape for example stem dim
            input_stems = input.unsqueeze(1)
            target_stems = target.unsqueeze(1)

        n_stems = input_stems.shape[1]

        # iterate over each stem compute loss for each transform
        for stem_idx in range(n_stems):
            input_stem = input_stems[:, stem_idx, ...]
            target_stem = target_stems[:, stem_idx, ...]

            # WIP: mask stems with very low energy
            # check energy of the stems before computing loss
            # input_energy = torch.mean(input_stem**2, dim=(-2, -1))
            # target_energy = torch.mean(target_stem**2, dim=(-2, -1))

            # create mask for stems if input or target are very low energy
            # mask = (input_energy > 1e-8) & (target_energy > 1e-8)

            # apply mask to remove batch items with low energy
            # masked_input_stem = input_stem[mask]
            # masked_target_stem = target_stem[mask]

            # if masked_input_stem.size(0) == 0:
            #    continue  # skip if all items in batch have low energy

            for transform, weight in zip(self.transforms, self.weights):
                transform_name = "_".join(transform.__name__.split("_")[1:])
                key = f"{self.sources_list[stem_idx]}-{transform_name}"
                input_transform = transform(input_stem, sample_rate=self.sample_rate)
                target_transform = transform(target_stem, sample_rate=self.sample_rate)
                val = torch.nn.functional.mse_loss(input_transform, target_transform)
                losses[key] = weight * val * self.source_weights[stem_idx]

        return losses


class ParameterEstimatorLoss(torch.nn.Module):
    def __init__(self, ckpt_path: str) -> None:
        super().__init__()
        # hard-coded model configuration
        self.encoder = SpectrogramEncoder(
            embed_dim=512,
            n_inputs=2,
            n_fft=2048,
            hop_length=512,
            input_batchnorm=False,
            encoder_batchnorm=True,
        )

        # read from checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        # extract parts of the state_dict only from the encoder
        state_dict = {}
        for key, val in ckpt["state_dict"].items():
            if key.startswith("encoder."):
                state_dict[key[8:]] = val

        print(state_dict)

        # load pretrained parameters
        self.encoder.load_state_dict(state_dict)

        # freeze the parameters in this model
        # for param in self.encoder.parameters():
        #    param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute loss on stereo mixes using pretrained parameter estimation encoder.

        Args:
            input: (bs, 2, seq_len)
            target: (bs, 2, seq_len)

        Returns:
            loss
        """
        # generate embeddings from input and target
        input_embed = self.encoder(input)  # bs, embed_dim
        target_embed = self.encoder(target)  # bs, embed_dim

        # compute loss
        loss = torch.nn.functional.mse_loss(input_embed, target_embed)

        return loss


class StereoCLAPLoss(torch.nn.Module):
    def __init__(self, distance: Callable = torch.nn.functional.mse_loss):
        super().__init__()
        self.distance = distance

        # instatiate pretrained CLAP model
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt()  # download the default pretrained checkpoint.

        # freeze all parameters in model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Compute loss on stereo mixes using featues from pretrained CLAP model.

        Args:
            input: (bs, 2, seq_len)
            target: (bs, 2, seq_len)

        Returns:
            loss: (batch_size, )
        """
        bs, chs, seq_len = input.size()

        assert chs == 2, "Input must be stereo"

        losses = {}

        input_mid, input_side = compute_mid_side(input)
        target_mid, target_side = compute_mid_side(target)

        signals = {
            "mid": [input_mid, target_mid],
            "side": [input_side, target_side],
        }

        for key, (input_signal, target_signal) in signals.items():
            # compute embeddings
            input_embed = self.model.get_audio_embedding_from_data(
                x=input_signal, use_tensor=True
            )
            target_embed = self.model.get_audio_embedding_from_data(
                x=target_signal, use_tensor=True
            )

            # compute losses
            sub_loss = self.distance(input_embed, target_embed)
            losses[key] = sub_loss

        return losses


class FX_encoder_loss(torch.nn.Module):
    def __init__(
        self,
        distance: Callable = torch.nn.functional.mse_loss,
        audiofeatures=True,
        weights: list[float] = [1.0],
    ):
        super().__init__()
        self.distance = distance
        config_path = "/homes/ssv02/Diff-MST/configs/models/fx_encoder_mst.yaml"
        # load configuration file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        checkpoint_path = "/homes/ssv02/Diff-MST/data/FXencoder_ps.pt"
        self.ddp = True
        # self.embed_distance = torch.nn.CosineEmbeddingLoss(reduction = 'mean')
        self.embed_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # load model
        self.model = FXencoder(self.config["Effects_Encoder"]["default"])

        # load checkpoint
        checkpoint = torch.load(checkpoint_path)

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            # remove `module.` if the model was trained with DDP
            name = k[7:] if self.ddp else k
            new_state_dict[name] = v

        # load params
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

        # freeze all parameters in model
        for param in self.model.parameters():
            param.requires_grad = False

        def compute_fx_embeds(x: torch.Tensor):
            embed = self.model(x)
            return embed

        # weights = [0.1,0.001,1.0,1.0,0.1,100.0]
        self.weights = weights
        self.transforms = []

        if audiofeatures:
            self.audiofeatures = audiofeatures

            self.transforms = [
                compute_rms,
                compute_crest_factor,
                compute_stereo_width,
                compute_stereo_imbalance,
                compute_barkspectrum,
            ]

        self.transforms.append(compute_fx_embeds)

        assert len(self.transforms) == len(self.weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        losses = {}
        # compute embeddings
        # input_embed = self.model(input)
        # target_embed = self.model(target)

        # # compute losses
        # loss = self.distance(input_embed, target_embed)

        # return loss

        for transform, weight in zip(self.transforms, self.weights):
            transform_name = "_".join(transform.__name__.split("_")[1:])
            # print(transform_name)
            input_transform = transform(input)
            target_transform = transform(target)
            if transform_name == "fx_embeds":
                val = 1 - self.embed_similarity(
                    input_transform, target_transform
                ).mean().clamp(min=1e-8)
                # print(val)
            else:
                val = torch.nn.functional.mse_loss(input_transform, target_transform)
                # print(val)
            losses[transform_name] = weight * val

        return losses


class QualityLoss(torch.nn.Module):
    def __init__(
        self,
        ckpt_path: str,
    ) -> None:
        super().__init__()
        # hard-coded model configuration
        encoder = SpectrogramEncoder(
            embed_dim=512,
            n_inputs=1,
            input_batchnorm=False,
            encoder_batchnorm=False,
            l2_norm=True,
        )

        self.model = QualityEstimationSystem.load_from_checkpoint(
            ckpt_path, encoder=encoder
        )
        self.model.eval()
        self.model.freeze()

    def forward(self, input: torch.Tensor, *args, **kwargs):
        """Compute loss on stereo mixes using featues from quality model.

        Args:
            input: (bs, 2, seq_len)
        """
        logits = self.model(input)  # higher is better (high quality)
        return -logits


class FeatureAndQualityLoss(torch.nn.Module):
    def __init__(
        self,
        weights: List[float],
        sample_rate: int,
        quality_ckpt_path: str,
        quality_weight: float = 1.0,
        stem_separation: bool = False,
        use_clap: bool = False,
    ):
        super().__init__()
        self.feature_loss = AudioFeatureLoss(
            weights=weights,
            sample_rate=sample_rate,
            stem_separation=stem_separation,
            use_clap=use_clap,
        )
        self.quality_loss = QualityLoss(quality_ckpt_path)
        self.quality_weight = quality_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        feature_losses = self.feature_loss(input, target)
        quality_loss = self.quality_loss(input)
        feature_losses["quality"] = quality_loss * self.quality_weight
        return feature_losses


# if __name__ == "__main__":
#     import torchaudio
#     path = "/import/c4dm-datasets-ext/mtg-jamendo_wav/02/1012002.wav"

#     #input1, sr = torchaudio.load(path, channels_first = True, num_frames = 44100*10)

#     input1= torch.zeros(2,44100*10)
#     input2 = input1
#     #input2, sr = torchaudio.load(path, channels_first = True, num_frames = 44100*10, frame_offset = 44100*10)
#     print(input1.shape)
#     input1 = input1.unsqueeze(0)
#     input2 = input2.unsqueeze(0)
#     weights = [1.0, 0.001, 1.0, 1.0, 0.01 , 0.01]

#     loss = FX_encoder_loss(weights = weights)
#     losses = loss(input1, input2)
#     print(losses)
#     print(sum(losses.values()))
