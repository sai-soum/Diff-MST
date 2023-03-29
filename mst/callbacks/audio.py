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
        dataloader_idx,
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
                    )

    def log_audio(
        self,
        outputs,
        batch_idx: int,
        sample_rate: int,
        global_step: int,
        logger,
        n_fft: int = 4096,
        hop_length: int = 1024,
    ):
        if "ref_mix_a" in outputs:
            x = outputs["ref_mix_a"][batch_idx, ...].float()
            x /= x.abs().max()

            logger.experiment.add_audio(
                f"{batch_idx+1}/ref_mix_a",
                x[0:1, :],
                global_step,
                sample_rate=sample_rate,
            )

        if "ref_mix_b" in outputs:
            y = outputs["ref_mix_b"][batch_idx, ...].float()
            y /= y.abs().max()

            logger.experiment.add_audio(
                f"{batch_idx+1}/ref_mix_b",
                y[0:1, :],
                global_step,
                sample_rate=sample_rate,
            )

        if "pred_mix_a" in outputs:
            y_hat = outputs["pred_mix_a"][batch_idx, ...].float()
            y_hat /= y_hat.abs().max()

            logger.experiment.add_audio(
                f"{batch_idx+1}/pred_mix_a",
                y_hat[0:1, :],
                global_step,
                sample_rate=sample_rate,
            )
