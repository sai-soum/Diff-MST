import torch
import torchaudio
import os
import glob

def low_pass_audio(y, sr, cutoff_freq):
    # cutoff_freq: in Hz
    # y: waveform
    # sr: sample rate
    # cutoff_freq: cutoff frequency
    # cutoff_freq = 4000
    y = torchaudio.functional.lowpass_biquad(y, sr, cutoff_freq)
    return y

def high_pass_audio(y, sr, cutoff_freq):
    # cutoff_freq: in Hz
    # y: waveform
    # sr: sample rate
    # cutoff_freq: cutoff frequency
    # cutoff_freq = 4000
    y = torchaudio.functional.highpass_biquad(y, sr, cutoff_freq)
    return y

def band_pass_audio(y, sr, low_cutoff_freq, high_cutoff_freq):
    # cutoff_freq: in Hz
    # y: waveform
    # sr: sample rate
    # low_cutoff_freq: low cutoff frequency
    # high_cutoff_freq: high cutoff frequency
    # low_cutoff_freq = 4000
    # high_cutoff_freq = 8000
    y = torchaudio.functional.bandpass_biquad(y, sr, low_cutoff_freq, high_cutoff_freq)
    return y

def pan_left_audio(y, sr):
    # y: waveform
    # sr: sample rate
    if y.shape[0] != 2:
        raise ValueError("Audio must have 2 channels for panning.")

    # Apply extreme panning to the left channel
    panned_waveform = torch.zeros_like(y)
    panned_waveform[0] = y[0]  # Left channel remains unchanged
    panned_waveform[1] = y[1] * 0.1  # Decrease amplitude of right channel (adjust value as needed)
    return panned_waveform

def pan_right_audio(y, sr):
    # y: waveform
    # sr: sample rate
    if y.shape[0] != 2:
        raise ValueError("Audio must have 2 channels for panning.")
    
    # Apply extreme panning to the right channel
    panned_waveform = torch.zeros_like(y)
    panned_waveform[0] = y[0] * 0.1  # Decrease amplitude of left channel (adjust value as needed)
    panned_waveform[1] = y[1]  # Right channel remains unchanged
    return panned_waveform




if __name__ == "__main__":
    ref_audio_paths = ["/Users/svanka/Downloads//diffmst-examples/song1/ref/_Feel it all Around_ by Washed Out (Portlandia Theme)_01.wav",
                       "/Users/svanka/Downloads//diffmst-examples/song2/ref/The Dip - Paddle To The Stars (Lyric Video)_01.wav",
                        "/Users/svanka/Downloads//diffmst-examples/song3/ref/Architects - _Doomsday__01.wav"]
    
    ref_save_path = "outputs/ablation_ref_examples"
    os.makedirs(ref_save_path, exist_ok=True)

    for ref_audio_path in ref_audio_paths:
        print(os.path.basename(ref_audio_path)  + "...")

        save_path = os.path.join(ref_save_path, os.path.basename(ref_audio_path).replace(".wav", ""))
        os.makedirs(save_path, exist_ok=True)


        y, sr = torchaudio.load(ref_audio_path, backend="soundfile")
        
        # Apply low-pass filter
        y_low_pass = low_pass_audio(y, sr, 5000)
        torchaudio.save(os.path.join(save_path, os.path.basename(ref_audio_path).replace(".wav", "_low_pass.wav")), y_low_pass, sr)

        # Apply high-pass filter
        y_high_pass = high_pass_audio(y, sr, 4000)
        torchaudio.save(os.path.join(save_path, os.path.basename(ref_audio_path).replace(".wav", "_high_pass.wav")), y_high_pass, sr)

        # Apply band-pass filter
        y_band_pass = band_pass_audio(y, sr, 500, 8000)
        torchaudio.save(os.path.join(save_path, os.path.basename(ref_audio_path).replace(".wav", "_band_pass.wav")), y_band_pass, sr)

        # Pan left
        y_pan_left = pan_left_audio(y, sr)
        torchaudio.save(os.path.join(save_path, os.path.basename(ref_audio_path).replace(".wav", "_pan_left.wav")), y_pan_left, sr)

        # Pan right
        y_pan_right = pan_right_audio(y, sr)
        torchaudio.save(os.path.join(save_path, os.path.basename(ref_audio_path).replace(".wav", "_pan_right.wav")), y_pan_right, sr)


