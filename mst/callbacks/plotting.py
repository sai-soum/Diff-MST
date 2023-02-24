import io
import torch
import librosa
import PIL.Image
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from typing import Any
from torch.functional import Tensor
from torchvision.transforms import ToTensor
from sklearn.metrics import ConfusionMatrixDisplay


def plot_spectrograms(
    input: torch.Tensor,
    target: torch.Tensor,
    estimate: torch.Tensor,
    n_fft: int = 4096,
    hop_length: int = 1024,
    sample_rate: float = 48000,
    filename: Any = None,
):
    """Create a side-by-side plot of the attention weights and the spectrogram.
    Args:
        input (torch.Tensor): Input audio tensor with shape [1 x samples].
        target (torch.Tensor): Target audio tensor with shape [1 x samples].
        estimate (torch.Tensor): Estimate of the target audio with shape [1 x samples].
        n_fft (int, optional): Analysis FFT size.
        hop_length (int, optional): Analysis hop length.
        sample_rate (float, optional): Audio sample rate.
        filename (str, optional): If a filename is supplied, the plot is saved to disk.
    """
    # use librosa to take stft
    x_stft = librosa.stft(
        input.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    x_D = librosa.amplitude_to_db(
        np.abs(x_stft),
        ref=np.max,
    )

    y_stft = librosa.stft(
        target.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    y_D = librosa.amplitude_to_db(
        np.abs(y_stft),
        ref=np.max,
    )

    y_hat_stft = librosa.stft(
        estimate.view(-1).numpy(),
        n_fft=n_fft,
        hop_length=hop_length,
    )
    y_hat_D = librosa.amplitude_to_db(
        np.abs(y_hat_stft),
        ref=np.max,
    )

    fig, axs = plt.subplots(
        nrows=3,
        sharex=True,
        figsize=(7, 6),
    )

    x_img = librosa.display.specshow(
        x_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[0],
    )

    y_img = librosa.display.specshow(
        y_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[1],
    )

    y_hat_img = librosa.display.specshow(
        y_hat_D,
        y_axis="log",
        x_axis="time",
        sr=sample_rate,
        hop_length=hop_length,
        ax=axs[2],
    )

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig2img(fig)


def plot_confusion_matrix(e_hat, e, labels=None, filename=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    cm = ConfusionMatrixDisplay.from_predictions(
        e,
        e_hat,
        labels=np.arange(len(labels)),
        display_labels=labels,
    )
    cm.plot(ax=ax, xticks_rotation="vertical")

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=300)

    return fig2img(fig)


def fig2img(fig, dpi=120):
    """Convert a matplotlib figure to JPEG to be show in Tensorboard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg", dpi=dpi)
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")
    return image
