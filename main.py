# class MyLightningCLI(pl.LightningCLI):
#    def add_arguments_to_parser(self, parser):
#        parser.link_arguments(
#            "model.model.mix_console.num_control_params",
#            "model.model.controller.num_control_params",
#            apply_on="instantiate",
#        )

import torch
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.strategies import DDPStrategy


def cli_main():
    cli = LightningCLI(
        save_config_callback=None,
        trainer_defaults={
            "accelerator": "gpu",
            # "strategy": DDPStrategy(find_unused_parameters=True),
            "log_every_n_steps": 50,
        },
    )


if __name__ == "__main__":

    cli_main()
