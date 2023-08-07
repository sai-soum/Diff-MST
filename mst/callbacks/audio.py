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
            if "dict" in key:  # skip parameters
                continue

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

        # now try to log parameters
        for key, param_dict in outputs.items():
            if "dict" not in key:  # skip audio
                continue

            column_names = None
            rows = []
            for effect_name, effect_params in param_dict.items():
                for param_name, param_val in effect_params.items():
                    row = []
                    row_name = f"{effect_name}.{param_name}"
                    row.append(row_name)

                    if column_names is None:
                        column_names = ["parameter"]
                        for i in range(param_val.shape[1]):
                            column_names.append(f"{i}")

                    for i in range(param_val.shape[1]):
                        row.append(param_val[batch_idx, i].item())
                    rows.append(row)

            wandb_table = wandb.Table(data=rows, columns=column_names)
            logger.experiment.log({f"batch={batch_idx}_parameters": wandb_table})
