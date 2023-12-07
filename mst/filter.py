import math
import torch
import warnings

# https://github.com/pytorch/audio/blob/d9942bae249329bd8c8bf5c92f0f108595fcb84f/torchaudio/functional/functional.py#L495


def _create_triangular_filterbank(
    all_freqs: torch.Tensor,
    f_pts: torch.Tensor,
) -> torch.Tensor:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    return fb


# https://github.com/pytorch/audio/blob/d9942bae249329bd8c8bf5c92f0f108595fcb84f/torchaudio/prototype/functional/functional.py#L6


def _hz_to_bark(freqs: float, bark_scale: str = "traunmuller") -> float:
    r"""Convert Hz to Barks.

    Args:
        freqs (float): Frequencies in Hz
        bark_scale (str, optional): Scale to use: ``traunmuller``, ``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        barks (float): Frequency in Barks
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "schroeder", "traunmuller" or "wang".'
        )

    if bark_scale == "wang":
        return 6.0 * math.asinh(freqs / 600.0)
    elif bark_scale == "schroeder":
        return 7.0 * math.asinh(freqs / 650.0)
    # Traunmuller Bark scale
    barks = ((26.81 * freqs) / (1960.0 + freqs)) - 0.53
    # Bark value correction
    if barks < 2:
        barks += 0.15 * (2 - barks)
    elif barks > 20.1:
        barks += 0.22 * (barks - 20.1)

    return barks


def _bark_to_hz(barks: torch.Tensor, bark_scale: str = "traunmuller") -> torch.Tensor:
    """Convert bark bin numbers to frequencies.

    Args:
        barks (torch.Tensor): Bark frequencies
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        freqs (torch.Tensor): Barks converted in Hz
    """

    if bark_scale not in ["schroeder", "traunmuller", "wang"]:
        raise ValueError(
            'bark_scale should be one of "traunmuller", "schroeder" or "wang".'
        )

    if bark_scale == "wang":
        return 600.0 * torch.sinh(barks / 6.0)
    elif bark_scale == "schroeder":
        return 650.0 * torch.sinh(barks / 7.0)
    # Bark value correction
    if any(barks < 2):
        idx = barks < 2
        barks[idx] = (barks[idx] - 0.3) / 0.85
    elif any(barks > 20.1):
        idx = barks > 20.1
        barks[idx] = (barks[idx] + 4.422) / 1.22

    # Traunmuller Bark scale
    freqs = 1960 * ((barks + 0.53) / (26.28 - barks))

    return freqs


def _hz_to_octs(freqs, tuning=0.0, bins_per_octave=12):
    a440 = 440.0 * 2.0 ** (tuning / bins_per_octave)
    return torch.log2(freqs / (a440 / 16))


def barkscale_fbanks(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_barks: int,
    sample_rate: int,
    bark_scale: str = "traunmuller",
) -> torch.Tensor:
    r"""Create a frequency bin conversion matrix.

    .. devices:: CPU

    .. properties:: TorchScript

    .. image:: https://download.pytorch.org/torchaudio/doc-assets/bark_fbanks.png
        :alt: Visualization of generated filter bank

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_barks (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        bark_scale (str, optional): Scale to use: ``traunmuller``,``schroeder`` or ``wang``. (Default: ``traunmuller``)

    Returns:
        torch.Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_barks``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * barkscale_fbanks(A.size(-1), ...)``.

    """

    # freq bins
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate bark freq bins
    m_min = _hz_to_bark(f_min, bark_scale=bark_scale)
    m_max = _hz_to_bark(f_max, bark_scale=bark_scale)

    m_pts = torch.linspace(m_min, m_max, n_barks + 2)
    f_pts = _bark_to_hz(m_pts, bark_scale=bark_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if (fb.max(dim=0).values == 0.0).any():
        warnings.warn(
            "At least one bark filterbank has all zero values. "
            f"The value for `n_barks` ({n_barks}) may be set too high. "
            f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
        )

    return fb
