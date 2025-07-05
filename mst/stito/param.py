import torch
import itertools
import torchaudio
import pytorch_lightning as pl


from typing import List
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def get_classifier_logits(extractor, model, audio: torch.Tensor, sr: int):
    """Get the logits from a pretrained audio classifier (AST trained on AudioSet) for a given audio file."""
    bs, chs, seq_len = audio.shape

    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    audio = audio.mean(dim=1, keepdim=False)  # must be mono

    # chunk into a list of mono tensors
    audio_tensors = torch.chunk(audio, bs, dim=0)

    # convert to list of np arrays
    audio_tensors = [audio.squeeze().cpu().numpy() for audio in audio_tensors]

    # Extract features (this has to be done on CPU)
    outputs = extractor(audio_tensors, sampling_rate=16000, return_tensors="pt")
    features = outputs.input_values

    # move to device
    features = features.type_as(audio)

    with torch.no_grad():
        logits = model(features).logits

    # here is how you get the classes if you want that...
    # predicted_class_ids = torch.argmax(logits, dim=-1).item()
    # predicted_label = model.config.id2label[predicted_class_ids]

    return logits.squeeze()


class ParameterEstimator(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        lr: float,
        num_instances: int,
        num_presets: int = 0,
        num_adv_classes: int = 0,
        adv_logits_type: str = "dataset",
        adv_weight: float = 1.0,
        weight_decay: float = 0.001,
        max_epochs: int = 250,
        embed_mode: str = "blind",
        norm: str = None,
        optimize_discriminator_only: bool = False,
    ):
        """

        Args:
            encoder (torch.nn.Module): Encoder model
            lr (float): Learning rate
            num_instances (int): Number of unique plugin instances for classification.
            num_presets (int): Number of unique presets for classification.
            num_adv_classes (int): Number of adversarial classes. Defaults to 0.
            adv_logits_type (str, optional): Adversarial logits type. Choices are "classifier" or "dataset". Defaults to "dataset".
            adv_weight (float, optional): Adversarial weight. Defaults to 1.0.
            weight_decay (float): Weight decay
            max_epochs (int, optional): Maximum number of epochs. Defaults to 500.
            embed_mode (str, optional): Embedding mode. Choices are "blind", "diff", "concat". Defaults to "blind".
            norm (str, optional): Normalization mode. Choices are "none" or "L2". Defaults to None.
            optimize_discriminator_only (bool, optional): Optimize discriminator only. Defaults to False.
        """
        super().__init__()
        self.save_hyperparameters(ignore="encoder")
        self.automatic_optimization = False

        # Base model f(.)
        self.encoder = encoder

        if embed_mode == "blind":
            input_dim = self.encoder.embed_dim * 2
        elif embed_mode == "diff":
            input_dim = self.encoder.embed_dim * 2
        elif embed_mode == "concat":
            input_dim = 2 * self.encoder.embed_dim * 2
        else:
            raise ValueError(f"Invalid embed_mode {embed_mode}")

        self.instance_estimator = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2 * input_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2 * input_dim, num_instances),
        )

        print("num_presets", num_presets)

        if num_presets > 0:
            self.preset_estimator = torch.nn.Sequential(
                torch.nn.Linear(input_dim + num_instances, 2 * input_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2 * input_dim, num_presets),
            )

        if num_adv_classes > 0:
            self.discriminator = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 2 * input_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2 * input_dim, num_adv_classes),
            )

        if adv_logits_type == "classifier":
            self.extractor = AutoFeatureExtractor.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            self.hparams.num_adv_classes = len(self.model.config.id2label)

    def configure_optimizers(self):
        params_to_optimize = [
            {"params": self.encoder.parameters(), "lr": self.hparams.lr},
            {"params": self.instance_estimator.parameters(), "lr": self.hparams.lr},
            {"params": self.preset_estimator.parameters(), "lr": self.hparams.lr},
        ]
        if self.hparams.num_adv_classes > 0:
            opt_g = torch.optim.Adam(
                params_to_optimize,
            )
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=self.hparams.lr,
            )
            return [opt_g, opt_d], []
        else:
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
            return optimizer

    def common_step(self, batch, mode="train"):
        inputs, outputs, instance_index, preset_index, tar_index, content_logits = batch

        if mode == "train":
            if self.hparams.num_adv_classes > 0:
                optimizer_g, optimizer_d = self.optimizers()
            else:
                optimizer_g = self.optimizers()

            if not self.hparams.optimize_discriminator_only:
                self.toggle_optimizer(optimizer_g)

        # encode outputs
        output_mid_feats, output_side_feats = self.encoder(outputs)

        if self.hparams.norm == "L2":
            outputs_mid_feats = torch.nn.functional.normalize(
                output_mid_feats, p=2, dim=-1
            )
            outputs_side_feats = torch.nn.functional.normalize(
                output_side_feats, p=2, dim=-1
            )
        elif self.hparams.norm is None:
            pass
        else:
            raise ValueError(f"Invalid norm {self.hparams.norm}")

        # Encode all images
        if self.hparams.embed_mode != "blind":
            # encoder inputs
            inputs_mid_feats, input_side_feats = self.encoder(inputs)

            if self.hparams.norm == "L2":
                inputs_mid_feats = torch.nn.functional.normalize(
                    inputs_mid_feats, p=2, dim=-1
                )
                inputs_side_feats = torch.nn.functional.normalize(
                    input_side_feats, p=2, dim=-1
                )
            elif self.hparams.norm is None:
                pass
            else:
                raise ValueError(f"Invalid norm {self.hparams.norm}")

            if self.hparams.embed_mode == "diff":
                # take the difference between input and outputs
                final_mid_feats = inputs_mid_feats - outputs_mid_feats
                final_side_feats = input_side_feats - output_side_feats
            elif self.hparams.embed_mode == "concat":
                # concatenate input and outputs
                final_feats = torch.cat(
                    (
                        inputs_mid_feats,
                        outputs_mid_feats,
                        inputs_side_feats,
                        output_side_feats,
                    ),
                    dim=-1,
                )
        else:
            final_feats = torch.cat((outputs_mid_feats, outputs_side_feats), dim=-1)

        # compute loss on effect instance
        instance_logits = self.instance_estimator(final_feats)
        instance_loss = torch.nn.functional.cross_entropy(
            instance_logits, instance_index
        )
        loss = instance_loss

        self.log(
            f"{mode}_instance_loss",
            instance_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # make predictions for presets
        if self.hparams.num_presets > 0:
            # concatenate instance_pred and final features
            concat_feats = torch.cat((instance_logits, final_feats), dim=-1)

            # predict preset
            preset_pred = self.preset_estimator(concat_feats)
            preset_loss = torch.nn.functional.cross_entropy(preset_pred, preset_index)
            self.log(
                f"{mode}_preset_loss",
                preset_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            loss += preset_loss

        # compute adversarial loss (if applicable)
        if self.hparams.num_adv_classes > 0:
            # compute discriminator loss
            adv_logits = self.discriminator(final_feats)

            if self.hparams.adv_logits_type == "classifier":
                adv_loss = -torch.nn.functional.cross_entropy(
                    adv_logits, content_logits
                )
            elif self.hparams.adv_logits_type == "dataset":
                adv_loss = -torch.nn.functional.cross_entropy(adv_logits, tar_index)
            else:
                raise ValueError(
                    f"Invalid adv_logits_type {self.hparams.adv_logits_type}"
                )
            # loss is negated to perform gradient ascent
            self.log(
                f"{mode}_adv_loss",
                adv_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            loss += adv_loss

        if mode == "train" and not self.hparams.optimize_discriminator_only:
            self.manual_backward(loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)

        # Logging loss
        self.log(
            mode + "_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # compute instance prediction accuracy
        instance_pred = torch.argmax(instance_logits, dim=-1)
        instance_acc = torch.sum(instance_pred == instance_index).float() / len(
            instance_index
        )
        self.log(
            f"{mode}_instance_acc",
            instance_acc,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # compute preset prediction accuracy
        if self.hparams.num_presets > 0:
            preset_pred = torch.argmax(preset_pred, dim=-1)
            preset_acc = torch.sum(preset_pred == preset_index).float() / len(
                preset_index
            )
            self.log(
                f"{mode}_preset_acc",
                preset_acc,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # compute discriminator loss
        if self.hparams.num_adv_classes > 0:
            # train the discriminator model
            if mode == "train":
                self.toggle_optimizer(optimizer_d)

            adv_logits = self.discriminator(final_feats.detach())
            if self.hparams.adv_logits_type == "classifier":
                d_loss = torch.nn.functional.cross_entropy(adv_logits, content_logits)
            elif self.hparams.adv_logits_type == "dataset":
                d_loss = torch.nn.functional.cross_entropy(adv_logits, tar_index)
            else:
                raise ValueError(
                    f"Invalid adv_logits_type {self.hparams.adv_logits_type}"
                )
            # scale the loss by the adversarial weight
            d_loss *= self.hparams.adv_weight

            if mode == "train":
                self.manual_backward(d_loss)
                optimizer_d.step()
                optimizer_d.zero_grad()
                self.untoggle_optimizer(optimizer_d)

            self.log(
                f"{mode}_d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

            # compute discriminator accuracy
            adv_pred = torch.argmax(adv_logits, dim=-1)
            adv_acc = torch.sum(adv_pred == tar_index).float() / len(tar_index)
            self.log(
                f"{mode}_adv_acc",
                adv_acc,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        return loss, instance_logits, instance_index

    def training_step(self, batch, batch_idx):
        loss, instance_pred, instance_index = self.common_step(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, instance_pred, instance_index = self.common_step(batch, mode="val")
        return instance_pred, instance_index
