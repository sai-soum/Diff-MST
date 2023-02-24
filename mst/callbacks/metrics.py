from re import X
import numpy as np
import pytorch_lightning as pl




class LogAudioMetricsCallback(pl.callbacks.Callback):
    def __init__(
        self,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self.metrics = {
            "PESQi": PESQi(sample_rate),
            "MRSTFTi": MRSTFTi(),
            "SISDRi": SISDRi(),
        }

        self.outputs = []

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
            self.outputs.append(outputs)

    def on_validation_end(self, trainer, pl_module):
        y_hat_metrics = {
            "PESQi": [],
            "MRSTFTi": [],
            "SISDRi": [],
        }
        for output in self.outputs:
            for metric_name, metric in self.metrics.items():
                for batch_idx in range(len(output["y_hat"].shape)):
                    y_hat = output["y_hat"][batch_idx, ...]
                    x = output["x"][batch_idx, ...]
                    y = output["y"][batch_idx, ...]

                    try:
                        val = metric(y_hat, x, y)
                        y_hat_metrics[metric_name].append(val)
                    except Exception as e:
                        print(e)

        # log final mean metrics
        for metric_name, metric in y_hat_metrics.items():
            val = np.mean(metric)
            trainer.logger.experiment.add_scalar(
                f"metrics/estimated-{metric_name}", val, trainer.global_step
            )

        # clear outputs
        self.outputs = []
