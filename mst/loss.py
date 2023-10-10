import torch
import librosa
import laion_clap
from typing import Callable, List

from mst.filter import barkscale_fbanks


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
    rms = torch.sqrt(torch.mean(x**2, dim=-1))
    return rms


def compute_crest_factor(x: torch.Tensor, **kwargs):
    """Compute crest factor as ratio of peak to rms energy in dB.

    Args:
        x: (bs, 2, seq_len)

    """
    cf = 20 * torch.log10(
        torch.max(torch.abs(x), dim=-1)[0] / compute_rms(x).clamp(min=1e-8)
    )
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
        self.transforms = [
            compute_rms,
            compute_crest_factor,
            compute_stereo_width,
            compute_stereo_imbalance,
            compute_barkspectrum,
        ]
        self.weights = weights
        self.sample_rate = sample_rate
        self.stem_separation = stem_separation
        self.sources_list = ["mix"]
        self.source_weights = [1.0]

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


class StereoCLAPLoss(torch.nn.Module):
    def __init__(self, sum_and_diff: bool = False, distance: str = "l2"):
        super().__init__()
        self.sum_and_diff = sum_and_diff
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

        if self.sum_and_diff:
            # compute sum and diff of stereo channels
            input_sum = input[:, 0, :] + input[:, 1, :]
            input_diff = input[:, 0, :] - input[:, 1, :]
            target_sum = target[:, 0, :] + target[:, 1, :]
            target_diff = target[:, 0, :] - target[:, 1, :]

            # compute embeddings
            input_sum_embeddings = self.model.get_audio_embedding_from_data(
                x=input_sum, use_tensor=True
            )
            target_sum_embeddings = self.model.get_audio_embedding_from_data(
                x=target_sum, use_tensor=True
            )
            input_diff_embeddings = self.model.get_audio_embedding_from_data(
                x=input_diff, use_tensor=True
            )
            target_diff_embeddings = self.model.get_audio_embedding_from_data(
                x=target_diff, use_tensor=True
            )

            # compute losses
            if self.distance == "l2":
                sum_loss = torch.nn.functional.mse_loss(
                    input_sum_embeddings, target_sum_embeddings
                )
                diff_loss = torch.nn.functional.mse_loss(
                    input_diff_embeddings, target_diff_embeddings
                )
            elif self.distance == "l1":
                sum_loss = torch.nn.functional.l1_loss(
                    input_sum_embeddings, target_sum_embeddings
                )
                diff_loss = torch.nn.functional.l1_loss(
                    input_diff_embeddings, target_diff_embeddings
                )
            else:
                raise ValueError(f"Invalid distance {self.distance}")

            # compute total loss
            loss = (sum_loss + diff_loss) / 2

        else:
            # move channel dim to batch dim
            input = input.view(bs * 2, -1)
            target = target.view(bs * 2, -1)

            # compute embeddings
            input_embeddings = self.model.get_audio_embedding_from_data(
                x=input, use_tensor=True
            )
            target_embeddings = self.model.get_audio_embedding_from_data(
                x=target, use_tensor=True
            )

            # compute losses
            if self.distance == "l2":
                loss = torch.nn.functional.mse_loss(input_embeddings, target_embeddings)
            elif self.distance == "l1":
                loss = torch.nn.functional.l1_loss(input_embeddings, target_embeddings)
            else:
                raise ValueError(f"Invalid distance {self.distance}")

        return loss
