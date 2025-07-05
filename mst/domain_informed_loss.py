import torch
import librosa
import torch.nn as nn

from typing import List
from mst.filter import barkscale_fbanks # Assuming this is available and correct
from collections import defaultdict
import itertools

# Ensure these are imported from the correct places in your project
# from mst.fx_encoder import FXencoder
# from mst.modules import SpectrogramEncoder


def compute_mid_side(x: torch.Tensor):
    """
    Computes the mid (sum) and side (difference) signals from a stereo input.

    Args:
        x: (bs, 2, seq_len) - Batch of stereo audio signals.

    Returns:
        x_mid: (bs, seq_len) - Mid (sum) signal.
        x_side: (bs, seq_len) - Side (difference) signal.
    """
    assert x.dim() == 3 and x.shape[1] == 2, "Input to compute_mid_side must be (bs, 2, seq_len)"
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
        x: (bs, 2, seq_len) - Input stereo audio.
        sample_rate: sample rate of audio.
        fft_size: size of fft.
        n_bins: number of mel bins.

    Returns:
        X: (bs, n_bins) - Mel spectrogram, mean over time and channels.
    """
    fb = librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_bins)
    fb = torch.tensor(fb, dtype=x.dtype, device=x.device).unsqueeze(0)

    if x.dim() == 2:
        x_mono = x.unsqueeze(1)
    elif x.dim() == 3 and x.shape[1] == 2:
        x_mono = x.mean(dim=1, keepdim=True)
    elif x.dim() == 3 and x.shape[1] == 1:
        x_mono = x
    else:
        raise ValueError(f"Unsupported input shape for compute_melspectrum: {x.shape}. Expected (bs, seq_len) or (bs, 2, seq_len)")
    
    X = torch.fft.rfft(x_mono, n=fft_size, dim=-1)
    X = torch.abs(X)
    X = X.squeeze(1)
    
    X = torch.matmul(fb, X.unsqueeze(-1)).squeeze(-1)
    X = torch.log(X + 1e-8)

    return X


def compute_barkspectrum(
    x: torch.Tensor,
    fft_size: int = 32768,
    n_bands: int = 24, # This n_bands refers to the target number of bark bands per channel (mid/side/stereo)
    sample_rate: int = 44100,
    f_min: float = 20.0,
    f_max: float = 20000.0,
    mode: str = "mid-side",
    **kwargs,
):
    """Compute bark-spectrogram.

    Args:
        x: (bs, ch, seq_len) - Input audio.
        fft_size: size of fft.
        n_bands: number of bark bins.
        sample_rate: sample rate of audio.
        f_min: minimum frequency.
        f_max: maximum frequency.
        mode: "mono", "stereo", or "mid-side" to specify how channels are processed.

    Returns:
        X: (bs, n_bands) for mono, or (bs, 2 * n_bands) for stereo/mid-side.
    """
    # compute filterbank (n_bands_from_barkscale_fbanks, fft_size // 2 + 1)
    fb_np = barkscale_fbanks((fft_size // 2) + 1, f_min, f_max, n_bands, sample_rate)
    # --- CHANGE START ---
    # The `barkscale_fbanks` function seems to return (freq_bins, n_bands)
    # E.g., (16385, 24).
    # To apply it to (bs, freq_bins), we need `(bs, freq_bins) @ (freq_bins, n_bands)`.
    # So, we want fb to be (freq_bins, n_bands).
    # Unsqueeze(0) was adding a batch dim, making it (1, 16385, 24).
    # We need to explicitly handle this:
    fb_tensor = torch.tensor(fb_np, dtype=x.dtype, device=x.device) # Shape: (16385, 24)
    # --- CHANGE END ---

    signals = []
    if x.dim() == 2:
        if mode != "mono": raise ValueError("Mono input (bs, seq_len) cannot be processed in 'stereo' or 'mid-side' mode.")
        signals = [x]
    elif x.dim() == 3:
        if x.shape[1] == 1:
            if mode != "mono": raise ValueError("Mono input (bs, 1, seq_len) cannot be processed in 'stereo' or 'mid-side' mode.")
            signals = [x.squeeze(1)]
        elif x.shape[1] == 2:
            if mode == "mono":
                signals = [x.mean(dim=1)]
            elif mode == "stereo":
                signals = [x[:, 0, :], x[:, 1, :]]
            elif mode == "mid-side":
                x_mid, x_side = compute_mid_side(x)
                signals = [x_mid, x_side]
            else: raise ValueError(f"Invalid mode {mode}")
        else: raise ValueError(f"Unsupported input channel dimension for compute_barkspectrum: {x.shape}")
    else: raise ValueError(f"Unsupported input shape for compute_barkspectrum: {x.shape}")


    outputs = []
    for i, signal in enumerate(signals): # Each signal is (bs, seq_len)
        window = torch.hann_window(fft_size, device=x.device, dtype=x.dtype)
        X_stft = torch.stft(
            signal,
            n_fft=fft_size,
            hop_length=fft_size // 4,
            win_length=fft_size,
            window=window,
            center=True,
            return_complex=True,
        ) # (bs, fft_size // 2 + 1, n_frames) -> (bs, 16385, n_frames)

        X_magnitude = torch.abs(X_stft) # (bs, 16385, n_frames)
        X_mean_time = torch.mean(X_magnitude, dim=-1) # (bs, 16385)

        # --- CHANGE START ---
        # Apply filterbank
        # We need to multiply (bs, freq_bins) by (freq_bins, n_bands) to get (bs, n_bands)
        X_filtered = torch.matmul(X_mean_time, fb_tensor) # (bs, n_bands)
        # --- CHANGE END ---
        
        X_log = torch.log(X_filtered + 1e-8)
        outputs.append(X_log)

    X_final = torch.cat(outputs, dim=-1)

    return X_final

# ... (rest of the code - AudioFeatureLoss and FullDiffMSTLoss - remains unchanged from previous version) ...


def compute_rms(x: torch.Tensor, **kwargs):
    """Compute root mean square energy.

    Args:
        x: (bs, ch, seq_len) or (bs, seq_len) - Input audio.

    Returns:
        rms: (bs, ) - RMS energy per batch item.
    """
    if x.dim() == 3:
        rms = torch.sqrt(torch.mean(x**2, dim=[-1, -2]).clamp(min=1e-8))
    else: # assuming (bs, seq_len)
        rms = torch.sqrt(torch.mean(x**2, dim=-1).clamp(min=1e-8))
    return rms


def compute_crest_factor(x: torch.Tensor, **kwargs):
    """Compute crest factor as ratio of peak to rms energy in dB.

    Args:
        x: (bs, ch, seq_len) - Input audio.

    Returns:
        cf: (bs, ) - Crest factor per batch item in dB.
    """
    num = torch.max(torch.abs(x), dim=-1)[0]
    if x.dim() == 3: # If stereo, take max across channels too
        num = torch.max(num, dim=-1)[0]

    den = compute_rms(x).clamp(min=1e-8)
    cf = 20 * torch.log10((num / den).clamp(min=1e-8))
    return cf


def compute_stereo_width(x: torch.Tensor, **kwargs):
    """Compute stereo width as ratio of energy in sum and difference signals.

    Args:
        x: (bs, 2, seq_len) - Input stereo audio.

    Returns:
        stereo_width: (bs, ) - Stereo width per batch item.
    """
    bs, chs, seq_len = x.size()
    assert chs == 2, "Input must be stereo for compute_stereo_width"

    x_sum, x_diff = compute_mid_side(x)
    sum_energy = torch.mean(x_sum**2, dim=-1)
    diff_energy = torch.mean(x_diff**2, dim=-1)
    stereo_width = diff_energy / sum_energy.clamp(min=1e-8)
    return stereo_width


def compute_stereo_imbalance(x: torch.Tensor, **kwargs):
    """Compute stereo imbalance as ratio of energy in left and right channels.

    Args:
        x: (bs, 2, seq_len) - Input stereo audio.

    Returns:
        stereo_imbalance: (bs, ) - Stereo imbalance per batch item.
    """
    left_energy = torch.mean(x[:, 0, :] ** 2, dim=-1)
    right_energy = torch.mean(x[:, 1, :] ** 2, dim=-1)

    stereo_imbalance = (right_energy - left_energy) / (
        right_energy + left_energy
    ).clamp(min=1e-8)

    return stereo_imbalance


import torch
import torch.nn as nn
from typing import List

# Assume compute_rms, compute_crest_factor, compute_stereo_width,
# compute_stereo_imbalance, compute_barkspectrum are defined elsewhere,
# along with necessary imports like librosa, barkscale_fbanks, etc.

class AudioFeatureLoss(torch.nn.Module):
    def __init__(
        self,
        weights: List[float],
        sample_rate: int,
        stem_separation: bool = False,
        use_clap: bool = False,
    ) -> None:
        super().__init__()
        # self.weights = weights # Old way, uncomment if not using nn.Parameter
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
        # These are nn.Parameters, so they will be learned if optimiser includes them
        w_rms = torch.nn.Parameter(
            torch.tensor(weights[0], dtype=torch.float32, requires_grad=True)
        )
        w_cf = torch.nn.Parameter(
            torch.tensor(weights[1], dtype=torch.float32, requires_grad=True)
        )
        w_sw = torch.nn.Parameter(
            torch.tensor(weights[2], dtype=torch.float32, requires_grad=True)
        )
        w_si = torch.nn.Parameter(
            torch.tensor(weights[3], dtype=torch.float32, requires_grad=True)
        )
        w_bs = torch.nn.Parameter(
            torch.tensor(weights[4], dtype=torch.float32, requires_grad=True)
        )
        self.register_parameter("w_rms", w_rms)
        self.register_parameter("w_cf", w_cf)
        self.register_parameter("w_sw", w_sw)
        self.register_parameter("w_si", w_si)
        self.register_parameter("w_bs", w_bs)
        self.weights = [ # This list will hold the nn.Parameter objects
            self.w_rms,
            self.w_cf,
            self.w_sw,
            self.w_si,
            self.w_bs,
        ]

        assert len(self.transforms) == len(weights), \
            f"Number of transforms ({len(self.transforms)}) must match number of weights ({len(weights)})"

    # --- _normalize_transform_zscore function remains as defined in previous working solution ---
    def _normalize_transform_zscore(self, transform_tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Applies Z-score normalization (mean 0, std 1) to the last dimension of the tensor.
        Clamps standard deviation to prevent division by zero.
        Handles single-element tensors along the standardization dimension.
        """
        # (Debug prints within this function are commented out in your provided code,
        #  so they are omitted here for brevity.)

        # The key logic here handles the shape[-1] == 1 case by returning zeros_like.
        # This function will now ONLY be called if transform_tensor.shape[-1] > 1 due to the
        # conditional call in the forward method. So the 'if' block here is actually
        # unnecessary now, but harmless if left in.
        # For simplicity and correctness with how it's called now:
        
        mean_val = transform_tensor.mean(dim=-1, keepdim=True)
        std_val = transform_tensor.std(dim=-1, keepdim=True).clamp(min=epsilon)
        normalized_tensor = (transform_tensor - mean_val) / std_val
        return normalized_tensor
        

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        losses = {}
        input_audio = input
        target_audio = target

        for transform, weight in zip(self.transforms, self.weights):
            transform_name = "_".join(transform.__name__.split("_")[1:])
            key = f"{self.sources_list[0]}-{transform_name}"
            
            input_transform = transform(input_audio, sample_rate=self.sample_rate)
            target_transform = transform(target_audio, sample_rate=self.sample_rate)
            
            # print(f"DEBUG: {key} - input_transform {input_transform}, target_transform: {target_transform}")
            
            # --- Proactive NaN/Inf check after transform (important to keep) ---
            if torch.isnan(input_transform).any() or torch.isinf(input_transform).any():
                print(f"Warning: NaN/Inf detected in input_transform for {transform_name}. Assigning zero loss for this feature.")
                losses[key] = torch.tensor(0.0, device=input_audio.device)
                continue
            if torch.isnan(target_transform).any() or torch.isinf(target_transform).any():
                print(f"Warning: NaN/Inf detected in target_transform for {transform_name}. Assigning zero loss for this feature.")
                losses[key] = torch.tensor(0.0, device=input_audio.device)
                continue

            if input_transform.shape != target_transform.shape:
                raise ValueError(f"Shape mismatch for {transform_name}: "
                                 f"Input {input_transform.shape} vs Target {target_transform.shape}")

            # --- CHANGE START: Conditional Normalization in forward method ---
            # Apply Z-score normalization ONLY if the feature has more than one element along the last dimension.
            # Otherwise, use the raw values directly for MSE, similar to your old working code.
            if input_transform.shape[-1] > 1: # Applies to e.g., Bark Spectrum (shape [bs, 48])
                input_transform_norm = self._normalize_transform_zscore(input_transform)
                target_transform_norm = self._normalize_transform_zscore(target_transform)
                # print(f"DEBUG: {key} - Normalized input_transform_norm {input_transform_norm}, target_transform_norm: {target_transform_norm}")
            else: # Applies to scalar features like RMS, Crest Factor, Stereo Width, Stereo Imbalance (shape [bs, 1] or [bs])
                input_transform_norm = input_transform
                target_transform_norm = target_transform
                # print(f"DEBUG: {key} - Using RAW input_transform {input_transform_norm}, target_transform: {target_transform_norm} (no Z-score as scalar)")
            # --- CHANGE END ---

            val = torch.nn.functional.mse_loss(input_transform_norm, target_transform_norm)
            # print(f"DEBUG: {key} - MSE loss value: {val.item()}")
            losses[key] = weight * val * self.source_weights[0]
            print(f"DEBUG: {key} - Weighted loss: {losses[key].item()}")

        return losses, self.weights

class FullDiffMSTLoss(nn.Module):
    def __init__(
        self,
        af_loss_fn: torch.nn.Module,
        sample_rate=44100,
        lambda_energy=0.5,
        lambda_band=0.5,
        lambda_low_center=1.0,
        lambda_high_widen=1.0,
        lambda_vocal_center=1.0,
        lambda_double_pan=1.0,
        energy_threshold=0.01,
        low_band_indices=[0, 1, 2], # Assuming these map to low frequencies in bark scale
        high_band_indices=[18, 19, 20, 21, 22, 23], # Assuming these map to high frequencies in bark scale
        use_energy_preservation=True,
        use_band_preservation=True,
        use_low_band_centering=True,
        use_high_band_widening=True,
        use_vocal_centering=True,
        use_doubletrack_panning=True,
        curriculum_scale=1.0,
    ):
        super().__init__()
        self.af_loss_fn = af_loss_fn
        self.sample_rate = sample_rate
        # lambda_energy: Weight for the energy preservation loss on individual wet tracks.
        # Range: 0.0 (no effect) to 10.0+ (strong penalty for tracks becoming too quiet).
        # Influence: Higher values enforce a minimum perceived loudness for each track, preventing silence.
        self.lambda_energy = lambda_energy
        # lambda_band: Weight for the band preservation loss on the predicted mix.
        # Range: 0.0 (no effect) to 10.0+ (strong penalty for low-band energy falling below threshold).
        # Influence: Higher values ensure the overall low-frequency content of the mix remains present and impactful.
        self.lambda_band = lambda_band
        # lambda_low_center: Weight for the low-band centering loss on the predicted mix.
        # Range: 0.0 (no effect) to 5.0+ (strong penalty for low-frequency stereo side energy).
        # Influence: Higher values force low-frequency content (kick, bass) towards mono/center, improving mix solidity and mono compatibility.
        self.lambda_low_center = lambda_low_center
        # lambda_high_widen: Weight for the high-band widening loss on the predicted mix.
        # Range: 0.0 (no effect) to 5.0+ (strong incentive for high-frequency stereo side energy).
        # Influence: Higher values encourage high-frequency content to be spread wider, enhancing perceived spaciousness and clarity.
        self.lambda_high_widen = lambda_high_widen
        # lambda_vocal_center: Weight for centering tracks identified as "vocal" by name.
        # Range: 0.0 (no effect) to 5.0+ (strong penalty for stereo imbalance in vocal tracks).
        # Influence: Higher values ensure main vocals remain prominently centered in the mix.
        self.lambda_vocal_center = lambda_vocal_center
        # lambda_double_pan: Weight for panning tracks identified as "double" by name and low dry correlation.
        # Range: 0.0 (no effect) to 5.0+ (strong penalty for unbalanced left/right energy).
        # Influence: Higher values encourage identified double-tracked elements to be spread evenly/widely across the stereo field.
        self.lambda_double_pan = lambda_double_pan
        # energy_threshold: Minimum RMS energy threshold (linear scale) for `lambda_energy` and `lambda_band` losses.
        # Range: Small positive float, e.g., 0.001 to 0.1.
        # Influence: Defines what level is considered "too quiet" for penalization. Higher values mean tracks need to be louder to avoid penalty.
        self.energy_threshold = energy_threshold
        # low_band_indices: List of indices for Bark scale frequency bands representing low frequencies.
        # Influence: Defines the frequency region for low-end preservation and centering. Must correspond to the bark spectrum's bins.
        self.low_band_indices = low_band_indices
        # high_band_indices: List of indices for Bark scale frequency bands representing high frequencies.
        # Influence: Defines the frequency region for high-frequency widening. Must correspond to the bark spectrum's bins.
        self.high_band_indices = high_band_indices
        # use_energy_preservation: Boolean flag to enable/disable the energy preservation loss.
        # Influence: True to activate, False to deactivate.
        self.use_energy_preservation = use_energy_preservation
        # use_band_preservation: Boolean flag to enable/disable the band preservation loss.
        # Influence: True to activate, False to deactivate.
        self.use_band_preservation = use_band_preservation
        # use_low_band_centering: Boolean flag to enable/disable the low-band centering loss.
        # Influence: True to activate, False to deactivate.
        self.use_low_band_centering = use_low_band_centering
        # use_high_band_widening: Boolean flag to enable/disable the high-band widening loss.
        # Influence: True to activate, False to deactivate.
        self.use_high_band_widening = use_high_band_widening
        # use_vocal_centering: Boolean flag to enable/disable the vocal centering loss (based on track names).
        # Influence: True to activate, False to deactivate.
        self.use_vocal_centering = use_vocal_centering
        # use_doubletrack_panning: Boolean flag to enable/disable the double-track panning loss (based on track names and dry correlation).
        # Influence: True to activate, False to deactivate.
        self.use_doubletrack_panning = use_doubletrack_panning
        # curriculum_scale: Scaling factor for losses, useful for curriculum learning.
        self.curriculum_scale = curriculum_scale

    def compute_energy_preservation_loss(self, wet_tracks, curriculum_scale):
        # wet_tracks shape: (bs, ch, N, seq_len) - N is number of stems
        # --- CHANGE: Get N, ch from correct dimensions based on (bs, ch, N, seq_len) ---
        bs, ch, N, seq_len = wet_tracks.shape
        
        losses = []
        for i in range(N):
            # --- CHANGE: Select track as (bs, ch, seq_len) from (bs, ch, N, seq_len) ---
            track = wet_tracks[:, :, i, :] # (bs, ch, seq_len)
            rms = torch.sqrt(torch.mean(track ** 2, dim=[-1, -2]) + 1e-9) # RMS across channels and time
            penalty = torch.clamp(self.energy_threshold - rms, min=0.0)
            losses.append(penalty.mean()) # Mean across batch
        
        if not losses:
            return torch.tensor(0.0, device=wet_tracks.device)

        return self.lambda_energy * torch.stack(losses).mean() * curriculum_scale

    def compute_band_preservation_loss(self, pred_mix, curriculum_scale):
        # pred_mix shape: (bs, 2, seq_len)
        bark_spec = compute_barkspectrum(pred_mix, mode="mono", sample_rate=self.sample_rate) # (bs, n_bands)
        
        # Ensure indices are within bounds
        if not all(idx < bark_spec.shape[1] for idx in self.low_band_indices):
            raise IndexError(f"low_band_indices {self.low_band_indices} out of bark_spec bounds ({bark_spec.shape[1]} bands).")

        # Take exponential of log-bark-spectrum to get linear energy for thresholding
        band_energy = torch.exp(bark_spec[:, self.low_band_indices]) # (bs, len(low_band_indices))
        
        # Penalize if energy in these bands is below threshold
        band_penalty = torch.clamp(self.energy_threshold - band_energy, min=0.0).mean()
        
        return self.lambda_band * band_penalty * curriculum_scale

    def compute_low_band_centering_loss(self, pred_mix, curriculum_scale):
        # pred_mix shape: (bs, 2, seq_len)
        bark_ms = compute_barkspectrum(pred_mix, mode="mid-side", sample_rate=self.sample_rate) # (bs, 2 * n_bands)
        
        n_bands = bark_ms.shape[-1] // 2
        # Side bands are the second half of the bark_ms output
        side_bands = bark_ms[:, n_bands:] # (bs, n_bands)
        
        # Ensure indices are within bounds
        if not all(idx < side_bands.shape[1] for idx in self.low_band_indices):
            raise IndexError(f"low_band_indices {self.low_band_indices} out of side_bands bark spectrum bounds ({side_bands.shape[1]} bands).")

        # Energy in the side component of low bands
        low_side_energy = torch.exp(side_bands[:, self.low_band_indices]) # (bs, len(low_band_indices))
        
        # We want to minimize low_side_energy to center low frequencies
        return self.lambda_low_center * low_side_energy.mean() * curriculum_scale

    # def compute_high_band_widening_loss(self, pred_mix, curriculum_scale):
    #     # pred_mix shape: (bs, 2, seq_len)
    #     bark_ms = compute_barkspectrum(pred_mix, mode="mid-side", sample_rate=self.sample_rate) # (bs, 2 * n_bands)
        
    #     n_bands = bark_ms.shape[-1] // 2
    #     # Side bands are the second half of the bark_ms output
    #     side_bands = bark_ms[:, n_bands:] # (bs, n_bands)

    #     # Ensure indices are within bounds
    #     if not all(idx < side_bands.shape[1] for idx in self.high_band_indices):
    #         raise IndexError(f"high_band_indices {self.high_band_indices} out of side_bands bark spectrum bounds ({side_bands.shape[1]} bands).")

    #     # Energy in the side component of high bands
    #     high_side_energy = torch.exp(side_bands[:, self.high_band_indices]) # (bs, len(high_band_indices))
        
    #     # We want to encourage high_side_energy to be above a certain widening_target
    #     widening_target = 0.5 
    #     widen_penalty = torch.clamp(widening_target - high_side_energy, min=0.0).mean()
        
    #     # This loss penalizes if high-side energy is *below* the target, encouraging widening.
    #     return self.lambda_high_widen * widen_penalty * curriculum_scale
    def compute_high_band_widening_loss(self, pred_mix, curriculum_scale):
        # pred_mix shape: (bs, 2, seq_len)
        bark_ms = compute_barkspectrum(pred_mix, mode="mid-side", sample_rate=self.sample_rate) # (bs, 2 * n_bands)
        
        # --- ADD THESE DEBUG PRINTS ---
        # print(f"DEBUG IN compute_high_band_widening_loss:")
        # print(f"  bark_ms.shape: {bark_ms.shape}")
        # --- END DEBUG PRINTS ---

        n_bands = bark_ms.shape[-1] // 2
        # Side bands are the second half of the bark_ms output
        side_bands = bark_ms[:, n_bands:]
        
        # --- ADD THESE DEBUG PRINTS ---
        # print(f"  n_bands (calculated from bark_ms): {n_bands}")
        # print(f"  side_bands.shape: {side_bands.shape}")
        # print(f"  self.high_band_indices: {self.high_band_indices}")
        # --- END DEBUG PRINTS ---

        # Ensure indices are within bounds
        if not all(idx < side_bands.shape[1] for idx in self.high_band_indices):
            raise IndexError(f"high_band_indices {self.high_band_indices} out of side_bands bark spectrum bounds ({side_bands.shape[1]} bands).")

        # Energy in the side component of high bands
        high_side_energy = torch.exp(side_bands[:, self.high_band_indices]) # This is the line that's causing the assert.
        
        # We want to encourage high_side_energy to be above a certain widening_target
        widening_target = 0.5 
        widen_penalty = torch.clamp(widening_target - high_side_energy, min=0.0).mean()
        
        # This loss penalizes if high-side energy is *below* the target, encouraging widening.
        return self.lambda_high_widen * widen_penalty * curriculum_scale

    def compute_vocal_and_doubletrack_losses(self, wet_tracks: torch.Tensor, dry_tracks: torch.Tensor, track_names: List[str], curriculum_scale: float):
        # wet_tracks shape: (bs, ch_wet, N, seq_len)
        # dry_tracks shape: (bs, ch_dry, N, seq_len)
        bs, ch_wet, N_wet, seq_len_wet = wet_tracks.shape
        bs_dry, ch_dry, N_dry, seq_len_dry = dry_tracks.shape

        vocal_losses = []
        double_losses = []

        # Ensure the number of track names matches the number of stems
        if len(track_names) != N_wet:
            print(f"Warning: Number of track names ({len(track_names)}) does not match number of stems in wet_tracks ({N_wet}). Skipping vocal/double-track losses.")
            return torch.tensor(0.0, device=wet_tracks.device), torch.tensor(0.0, device=wet_tracks.device)
        
        # Ensure dry_tracks has compatible shape (bs, ch_dry, N, seq_len)
        if not (bs_dry == bs and N_dry == N_wet and seq_len_dry == seq_len_wet):
            print(f"Warning: Incompatible shape for dry_tracks ({dry_tracks.shape}) with wet_tracks ({wet_tracks.shape}). Skipping vocal/double-track losses.")
            return torch.tensor(0.0, device=wet_tracks.device), torch.tensor(0.0, device=wet_tracks.device)

        # --- VOCAL CENTERING LOGIC (remains largely the same) ---
        for i, track_name in enumerate(track_names):
            if self.use_vocal_centering and ("vocal" in track_name.lower()):
                wet_track = wet_tracks[:, :, i, :]
                if ch_wet == 2:
                    left_energy = torch.mean(wet_track[:, 0, :] ** 2, dim=-1)
                    right_energy = torch.mean(wet_track[:, 1, :] ** 2, dim=-1)
                    imbalance = (right_energy - left_energy) / (right_energy + left_energy + 1e-8)
                    vocal_losses.append(imbalance.abs().mean())
                else:
                    print(f"Warning: Track '{track_name}' named as 'vocal' but is not stereo. Skipping vocal centering loss for this track.")
        
        # --- DOUBLE-TRACKING PANNING LOGIC (REVISED) ---
        if self.use_doubletrack_panning:
            # Step 1: Group track indices by their lowercase name
            name_to_indices = defaultdict(list)
            for i, name in enumerate(track_names):
                name_to_indices[name.lower()].append(i)

            # Step 2: Iterate through groups that have more than one track
            # and form unique pairs within those groups.
            for name, indices_list in name_to_indices.items():
                if len(indices_list) < 2: # Need at least two tracks with the same name to form a pair
                    continue
                
                # Consider all unique pairs (combinations of 2) within this group
                for idx1, idx2 in itertools.combinations(indices_list, 2):
                    # Extract wet and dry tracks for this specific pair
                    track1_wet = wet_tracks[:, :, idx1, :]
                    track2_wet = wet_tracks[:, :, idx2, :]
                    track1_dry = dry_tracks[:, :, idx1, :]
                    track2_dry = dry_tracks[:, :, idx2, :]

                    # Check if both tracks in the pair are stereo (essential for stereo width/panning)
                    if ch_wet != 2 or ch_dry != 2:
                        print(f"Warning: Track pair '{track_names[idx1]}' and '{track_names[idx2]}' (same name: '{name}') are not fully stereo ({ch_wet} wet ch, {ch_dry} dry ch). Skipping double-track panning loss for this pair.")
                        continue
                    
                    # --- Condition: Dry tracks are sufficiently decorrelated ---
                    # Compute inter-track correlation on the MONO versions of the dry signals.
                    track1_dry_mono = track1_dry.mean(dim=1) # (bs, seq_len)
                    track2_dry_mono = track2_dry.mean(dim=1) # (bs, seq_len)

                    norm_t1_dry = torch.norm(track1_dry_mono, dim=-1)
                    norm_t2_dry = torch.norm(track2_dry_mono, dim=-1)
                    
                    # Add epsilon to denominator for stability for silent tracks
                    denominator_corr = (norm_t1_dry * norm_t2_dry) + 1e-8 
                    
                    # Calculate correlation. Clamp it to avoid numerical issues near +/-1
                    correlation_between_dry_tracks = (torch.sum(track1_dry_mono * track2_dry_mono, dim=-1) / denominator_corr).clamp(-1.0, 1.0)
                    
                    # Apply loss only if correlation is below a certain threshold (e.g., < 0.7 or lower like 0.5)
                    # Lower threshold means higher decorrelation is required for the loss to apply.
                    # We care about the magnitude of correlation, so use .abs()
                    correlation_threshold = 0.7 # Hyperparameter to tune (0.7 is a starting point, might need lower like 0.5 or 0.3 for true double-tracking)
                    is_decorrelated_mask_pair = (correlation_between_dry_tracks.abs() < correlation_threshold) 
                    
                    if is_decorrelated_mask_pair.any(): # Apply loss for batch items where condition is met
                        # --- Loss Objective for Double-Tracked Pairs: Encourage Widening ---
                        # Create a "pseudo-mix" from the wet pair for width calculation
                        combined_wet_pair = track1_wet + track2_wet # (bs, 2, seq_len)
                        
                        # Compute stereo width of the combined pair for relevant batch items
                        masked_combined_wet_pair = combined_wet_pair[is_decorrelated_mask_pair]

                        if masked_combined_wet_pair.numel() > 0:
                            pair_stereo_width = compute_stereo_width(masked_combined_wet_pair) # (num_masked_bs,)
                            
                            # We want the stereo width to be high (e.g., close to 1.0)
                            target_width = 0.95 # This is a hyperparameter for desired stereo width
                            
                            # Loss is higher if actual width is far from target width (from below)
                            width_penalty = torch.mean(torch.clamp(target_width - pair_stereo_width, min=0.0))
                            
                            double_losses.append(width_penalty)
                    else:
                        # Optional: Print why a pair was skipped if its dry correlation was too high
                        # print(f"Info: Skipping double-track loss for '{track_names[idx1]}' and '{track_names[idx2]}' (corr: {correlation_between_dry_tracks.mean().item():.2f}) - not decorrelated enough.")
                        pass # No loss if dry correlation is too high

        # Sum up losses, handling empty lists gracefully
        vocal_loss = torch.tensor(0.0, device=wet_tracks.device)
        if vocal_losses:
            vocal_loss = self.lambda_vocal_center * torch.stack(vocal_losses).mean() * curriculum_scale
        
        double_loss = torch.tensor(0.0, device=wet_tracks.device)
        if double_losses:
            double_loss = self.lambda_double_pan * torch.stack(double_losses).mean() * curriculum_scale
        
        return vocal_loss, double_loss


    def forward(
        self,
        pred_mix: torch.Tensor, # (bs, 2, seq_len)
        ref_mix: torch.Tensor,  # (bs, 2, seq_len)
        # --- CHANGE: Updated wet_tracks and dry_tracks docstrings to new dimension ---
        wet_tracks: torch.Tensor, # (bs, ch, N, seq_len) - N is number of stems (processed by your model), ch is num channels
        dry_tracks: torch.Tensor, # (bs, ch_dry, N, seq_len) - N is number of stems (unprocessed, original), ch_dry can be 1 or 2
        track_names: List[str], # List of N strings, corresponding to each stem in wet_tracks/dry_tracks
        global_step: int = None,
        max_step: int = None,
    ):
        """
        Computes the total loss for mixing style transfer, including audio features
        and domain-specific mixing rules.

        Args:
            pred_mix: (bs, 2, seq_len) - The predicted mixed audio.
            ref_mix: (bs, 2, seq_len) - The reference mixed audio (for style transfer).
            wet_tracks: (bs, ch, N, seq_len) - The individual wet (processed) tracks
                        before mixing, used for domain knowledge application.
            dry_tracks: (bs, ch_dry, N, seq_len) - The individual dry (unprocessed) tracks,
                        used for robust identification of certain track types.
            track_names: List[str] - A list of names for each stem in wet_tracks/dry_tracks.
            global_step: Current training step (for curriculum learning).
            max_step: Maximum training steps (for curriculum learning).

        Returns:
            A tuple containing:
            - total_loss: Sum of all weighted loss components.
            - total_style_loss: Sum of audio feature losses.
            - energy_preservation_loss: Loss component for energy preservation.
            - band_preservation_loss: Loss component for low-frequency band preservation.
            - low_band_centering_loss: Loss component for centering low frequencies.
            - high_band_widening_loss: Loss component for widening high frequencies.
            - vocal_loss: Loss component for centering probable vocals.
            - double_loss: Loss component for panning probable double-tracked signals.
        """
        # --- CHANGE: Removed previous unsqueeze logic based on new understanding of dry_tracks_shape ---
        # With (bs, ch, N, seq_len), dry_tracks should already have its channel dim.
        # No unsqueeze is needed here; shape validation is done in compute_vocal_and_doubletrack_losses.
        dry_tracks = dry_tracks.unsqueeze(1) if dry_tracks.dim() == 3 else dry_tracks # Ensure (bs, N, seq_len)

        # print("pred_mix_shape:", pred_mix.shape)
        # print("ref_mix_shape:", ref_mix.shape)
        # print("wet_tracks_shape:", wet_tracks.shape)
        # print("dry_tracks_shape:", dry_tracks.shape)
        # print("track_names:", track_names)
        # track_names: [('vox',), ('backingvox',), ('fiddle',), ('guitar',), ('guitar',), ('fiddle',), ('backingvox',), ('backingvox',)]
        # remove parantheses and convert to list of strings
        track_names = [name[0] if isinstance(name, tuple) else name for name in track_names]
        # print("track_names:", track_names)
        total_loss = torch.tensor(0.0, device=pred_mix.device)
        
        # 1. Style Similarity Loss (AudioFeatureLoss)
        style_loss_dict, learnt_af_weights = self.af_loss_fn(pred_mix, ref_mix)
        # sum only the absolute values of the style losses after meaning them
        
        total_style_loss = sum(torch.abs(val.mean()) for val in style_loss_dict.values())
        # print(f"Style loss: {total_style_loss.item():.4f}")
        total_loss += total_style_loss

        # Curriculum scaling for domain knowledge losses
        curriculum_scale = self.curriculum_scale
        if global_step is not None and max_step is not None and max_step > 0:
            curriculum_scale = min(global_step / max_step, 1.0)
        # print(f"Curriculum Scale: {curriculum_scale:.4f}")

        # 2. Domain Knowledge Losses
        energy_preservation_loss = torch.tensor(0.0, device=pred_mix.device)
        if self.use_energy_preservation:
            energy_preservation_loss = self.compute_energy_preservation_loss(wet_tracks, curriculum_scale)
            total_loss += energy_preservation_loss
        # print(f"Energy preservation loss: {energy_preservation_loss.item():.4f}")
            
        band_preservation_loss = torch.tensor(0.0, device=pred_mix.device)
        if self.use_band_preservation:
            band_preservation_loss = self.compute_band_preservation_loss(pred_mix, curriculum_scale)
            total_loss += band_preservation_loss
        # print(f"Band preservation loss: {band_preservation_loss.item():.4f}")
            
        low_band_centering_loss = torch.tensor(0.0, device=pred_mix.device)
        if self.use_low_band_centering:
            low_band_centering_loss = self.compute_low_band_centering_loss(pred_mix, curriculum_scale)
            total_loss += low_band_centering_loss
        # print(f"Low band centering loss: {low_band_centering_loss.item():.4f}")
            
        high_band_widening_loss = torch.tensor(0.0, device=pred_mix.device)
        if self.use_high_band_widening:
            high_band_widening_loss = self.compute_high_band_widening_loss(pred_mix, curriculum_scale)
            total_loss += high_band_widening_loss
        # print(f"High band widening loss: {high_band_widening_loss.item():.4f}")
        
        vocal_loss = torch.tensor(0.0, device=pred_mix.device)
        double_loss = torch.tensor(0.0, device=pred_mix.device)

        if self.use_vocal_centering or self.use_doubletrack_panning:
            # Pass both wet_tracks and dry_tracks to the computation function
            vocal_loss, double_loss = self.compute_vocal_and_doubletrack_losses(wet_tracks, dry_tracks, track_names, curriculum_scale)
            
            if self.use_vocal_centering:
                total_loss += vocal_loss
            if self.use_doubletrack_panning:
                total_loss += double_loss
                
        # print(f"Vocal loss: {vocal_loss.item():.4f}")
        # print(f"Double track loss: {double_loss.item():.4f}")

        # return total_loss
        
        return (
            total_loss,
            style_loss_dict,
            total_style_loss,
            energy_preservation_loss,
            band_preservation_loss,
            low_band_centering_loss,
            high_band_widening_loss,
            vocal_loss,
            double_loss,
            learnt_af_weights,
        )