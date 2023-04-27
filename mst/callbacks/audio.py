import torch
import wandb
import numpy as np
import pytorch_lightning as pl


from mst.callbacks.plotting import plot_spectrograms


class LogAudioCallback(pl.callbacks.Callback):
    def __init__(
        self,
        num_examples: int = 8,
        peak_normalize: bool = True,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.num_examples = num_examples
        self.peak_normalize = peak_normalize
        self.sample_rate = sample_rate

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        
    ):
        """Called when the validation batch ends."""
        if outputs is not None:
            num_examples = outputs["ref_mix_a"].shape[0]
            if num_examples > self.num_examples:
                num_examples = self.num_examples

            if batch_idx == 0:
                for n in range(num_examples):
                    self.log_audio(
                        outputs,
                        n,
                        pl_module.model.mix_console.sample_rate,
                        trainer.global_step,
                        trainer.logger,
                        f"Epoch {trainer.current_epoch}",
                    )

    def log_audio(
        self,
        outputs,
        batch_idx: int,
        sample_rate: int,
        global_step: int,
        logger,
        caption: str,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        audio_files = []
        audio_keys = []
        total_samples = 0
        # put all audio in file
        for key, audio in outputs.items():
            x = audio[batch_idx, ...].float()
            x = x.permute(1, 0)
            x /= x.abs().max()
            audio_files.append(x)
            audio_keys.append(key)
            total_samples += x.shape[0]

        y = torch.zeros(total_samples + int(len(audio_keys) * sample_rate), 2)
        name = f"{batch_idx}_"
        start = 0
        for x, key in zip(audio_files, audio_keys):
            end = start + x.shape[0]
            y[start:end, :] = x
            start = end + int(sample_rate)
            name += key + "-"

        logger.experiment.log(
            {
                f"{name}": wandb.Audio(
                    y.numpy(),
                    caption=caption,
                    sample_rate=int(sample_rate),
                )
            }
        )
