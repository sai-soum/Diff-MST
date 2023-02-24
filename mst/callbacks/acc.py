import torch
import numpy as np
import pytorch_lightning as pl

from mst.callbacks.plotting import plot_confusion_matrix


class ConfusionMatrixCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.estimates = []

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
            self.targets.append(outputs["e"])
            self.estimates.append(outputs["e_hat"].max(1).indices)

    def on_validation_end(self, trainer, pl_module):

        e = torch.cat(self.targets, dim=0)
        e_hat = torch.cat(self.estimates, dim=0)

        trainer.logger.experiment.add_image(
            f"confusion_matrix",
            plot_confusion_matrix(
                e_hat,
                e,
                labels=pl_module.hparams.effects,
            ),
            trainer.global_step,
        )

        # clear outputs
        self.targets = []
        self.estimates = []
