import torch
import librosa
import laion_clap
from typing import Callable, List


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
    ) -> None:
        """Compute loss using a set of differentiable audio features.

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
            compute_melspectrum,
        ]
        self.weights = weights
        self.sample_rate = sample_rate
        assert len(self.transforms) == len(weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        losses = {}
        for transform, weight in zip(self.transforms, self.weights):
            input_transform = transform(
                input,
                sample_rate=self.sample_rate,
            )
            target_transform = transform(target)
            val = torch.nn.functional.mse_loss(input_transform, target_transform)
            losses[transform.__name__] = weight * val

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
