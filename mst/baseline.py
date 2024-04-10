# adapted from https://github.com/jhtonyKoo/music_mixing_style_transfer
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torchaudio
import pyloudnorm as pyln

from collections import OrderedDict


def equal_loudness_mix(tracks: torch.Tensor, *args, **kwargs):

    meter = pyln.Meter(44100)
    target_lufs_db = -48.0

    norm_tracks = []
    for track_idx in range(tracks.shape[1]):
        track = tracks[:, track_idx : track_idx + 1, :]
        lufs_db = meter.integrated_loudness(track.squeeze(0).permute(1, 0).numpy())

        if lufs_db < -80.0:
            print(f"Skipping track {track_idx} with {lufs_db:.2f} LUFS.")
            continue

        lufs_delta_db = target_lufs_db - lufs_db
        track *= 10 ** (lufs_delta_db / 20)
        norm_tracks.append(track)

    norm_tracks = torch.cat(norm_tracks, dim=1)
    # create a sum mix with equal loudness
    sum_mix = torch.sum(norm_tracks, dim=1, keepdim=True).repeat(1, 2, 1)
    sum_mix /= sum_mix.abs().max()

    return sum_mix, None, None, None


def load_mixing_style_transfer_model(
    ckpt_dir: str = "checkpoints/mst/", ddp: bool = True
):

    fx_encoder_ckpt_path = f"{ckpt_dir}/FXencoder_ps.pt"
    mixing_converter_ckpt_path = f"{ckpt_dir}/MixFXcloner_ps.pt"

    # download models via GDrive from https://github.com/jhtonyKoo/music_mixing_style_transfer

    # model architecture configurations
    fx_encoder_config = {
        "channels": [16, 32, 64, 128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
        "kernels": [25, 25, 15, 15, 10, 10, 10, 10, 5, 5, 5, 5],
        "strides": [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
        "dilation": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "bias": True,
        "norm": "batch",
        "conv_block": "res",
        "activation": "relu",
    }

    condition_dimension = 2048
    nblocks = 14
    dilation_growth = 2
    kernel_size = 15
    channel_width = 128
    stack_size = 15
    causal = False

    # instantiate the models
    models = {}
    models["effects_encoder"] = FXencoder(fx_encoder_config)
    models["mixing_converter"] = TCNModel(
        nparams=condition_dimension,
        ninputs=2,
        noutputs=2,
        nblocks=nblocks,
        dilation_growth=dilation_growth,
        kernel_size=kernel_size,
        channel_width=channel_width,
        stack_size=stack_size,
        cond_dim=condition_dimension,
        causal=causal,
    )

    # load fx encoder checkpoint
    fx_encoder_ckpt = torch.load(fx_encoder_ckpt_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in fx_encoder_ckpt["model"].items():
        # remove `module.` if the model was trained with DDP
        name = k[7:] if ddp else k
        new_state_dict[name] = v

    # load params
    models["effects_encoder"].load_state_dict(new_state_dict)
    models["effects_encoder"].eval()

    # load mixing converter checkpoint
    mixing_converter_ckpt = torch.load(mixing_converter_ckpt_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in mixing_converter_ckpt["model"].items():
        # remove `module.` if the model was trained with DDP
        name = k[7:] if ddp else k
        new_state_dict[name] = v

    # load params
    models["mixing_converter"].load_state_dict(new_state_dict)
    models["mixing_converter"].eval()

    return models


def run_mixing_style_transfer_model(
    tracks: torch.Tensor,
    ref_mix: torch.Tensor,
    models,
    *args,
    **kwargs,
):
    separation_model = "mdx_extra"
    separation_device = "cpu"
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    stem_names = ["bass", "drums", "other", "vocals"]

    # temporarily save ref_mix to disk
    ref_filename = "ref_mix"
    ref_mix_file_path = os.path.join(tmp_dir, f"{ref_filename}.wav")
    torchaudio.save(ref_mix_file_path, ref_mix.view(2, -1), 44100)

    # run demucs on the ref_mix
    cmd_line = f"demucs {ref_mix_file_path} -n {separation_model} -d {separation_device} -o {tmp_dir}"
    os.system(cmd_line)

    # load the separate tracked into memory
    ref_stems = {}
    for stem_name in stem_names:
        stem_filepath = os.path.join(
            tmp_dir, separation_model, ref_filename, f"{stem_name}.wav"
        )
        x, sr = torchaudio.load(stem_filepath)
        ref_stems[stem_name] = x

    # sum the tracks and separate them into stems
    equal_loud_mix = equal_loudness_mix(tracks)
    equal_loud_mix = equal_loud_mix[0].squeeze(0)
    equal_loud_mix_filepath = os.path.join(tmp_dir, "tracks.wav")
    torchaudio.save(equal_loud_mix_filepath, equal_loud_mix.view(2, -1), 44100)

    # run demucs on the tracks that have been summed
    cmd_line = f"demucs {equal_loud_mix_filepath} -n {separation_model} -d {separation_device} -o {tmp_dir}"
    os.system(cmd_line)

    # load each separated stem
    track_stems = {}
    for stem_name in stem_names:
        stem_filepath = os.path.join(
            tmp_dir, separation_model, "tracks", f"{stem_name}.wav"
        )
        x, sr = torchaudio.load(stem_filepath, backend="soundfile")
        track_stems[stem_name] = x

    # extract features from reference stem and mix associated stem
    mixed_stems = []
    for stem_name in stem_names:
        reference_feature = models["effects_encoder"](ref_stems[stem_name].unsqueeze(0))
        print(reference_feature.shape)
        infered_ref_data_avg = torch.mean(reference_feature, dim=0)
        infered_data = models["mixing_converter"](
            track_stems[stem_name].unsqueeze(0), infered_ref_data_avg.unsqueeze(0)
        )
        mixed_stems.append(infered_data)

    # sum the mixed stems
    mixed_stems = torch.stack(mixed_stems)
    mixed_stems = torch.sum(mixed_stems, dim=0)

    return mixed_stems, None, None, None


# FXencoder that extracts audio effects from music recordings trained with a contrastive objective
class FXencoder(nn.Module):
    def __init__(self, config):
        super(FXencoder, self).__init__()
        # input is stereo channeled audio
        config["channels"].insert(0, 2)

        # encoder layers
        encoder = []
        for i in range(len(config["kernels"])):
            if config["conv_block"] == "res":
                encoder.append(
                    Res_ConvBlock(
                        dimension=1,
                        in_channels=config["channels"][i],
                        out_channels=config["channels"][i + 1],
                        kernel_size=config["kernels"][i],
                        stride=config["strides"][i],
                        padding="SAME",
                        dilation=config["dilation"][i],
                        norm=config["norm"],
                        activation=config["activation"],
                        last_activation=config["activation"],
                    )
                )
            elif config["conv_block"] == "conv":
                encoder.append(
                    ConvBlock(
                        dimension=1,
                        layer_num=1,
                        in_channels=config["channels"][i],
                        out_channels=config["channels"][i + 1],
                        kernel_size=config["kernels"][i],
                        stride=config["strides"][i],
                        padding="VALID",
                        dilation=config["dilation"][i],
                        norm=config["norm"],
                        activation=config["activation"],
                        last_activation=config["activation"],
                        mode="conv",
                    )
                )
        self.encoder = nn.Sequential(*encoder)

        # pooling method
        self.glob_pool = nn.AdaptiveAvgPool1d(1)

    # network forward operation
    def forward(self, input):
        enc_output = self.encoder(input)
        glob_pooled = self.glob_pool(enc_output).squeeze(-1)

        # outputs c feature
        return glob_pooled


# MixFXcloner which is based on a Temporal Convolutional Network (TCN) module
# original implementation : https://github.com/csteinmetz1/micro-tcn
import pytorch_lightning as pl


class TCNModel(pl.LightningModule):
    """Temporal convolutional network with conditioning module.
    Args:
        nparams (int): Number of conditioning parameters.
        ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
        noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
        nblocks (int): Number of total TCN blocks. Default: 10
        kernel_size (int): Width of the convolutional kernels. Default: 3
        dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
        channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 2
        channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
        stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
        grouped (bool): Use grouped convolutions to reduce the total number of parameters. Default: False
        causal (bool): Causal TCN configuration does not consider future input values. Default: False
        skip_connections (bool): Skip connections from each block to the output. Default: False
        num_examples (int): Number of evaluation audio examples to log after each epochs. Default: 4
    """

    def __init__(
        self,
        nparams,
        ninputs=1,
        noutputs=1,
        nblocks=10,
        kernel_size=3,
        dilation_growth=1,
        channel_growth=1,
        channel_width=32,
        stack_size=10,
        cond_dim=2048,
        grouped=False,
        causal=False,
        skip_connections=False,
        num_examples=4,
        save_dir=None,
        **kwargs,
    ):
        super(TCNModel, self).__init__()
        self.save_hyperparameters()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs

            if self.hparams.channel_growth > 1:
                out_ch = in_ch * self.hparams.channel_growth
            else:
                out_ch = self.hparams.channel_width

            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            self.blocks.append(
                TCNBlock(
                    in_ch,
                    out_ch,
                    kernel_size=self.hparams.kernel_size,
                    dilation=dilation,
                    padding="same" if self.hparams.causal else "valid",
                    causal=self.hparams.causal,
                    cond_dim=cond_dim,
                    grouped=self.hparams.grouped,
                    conditional=True if self.hparams.nparams > 0 else False,
                )
            )

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, cond):
        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            # for SeFa
            if isinstance(cond, list):
                x = block(x, cond[idx])
            else:
                x = block(x, cond)
            skips = 0

        out = torch.clamp(self.output(x + skips), min=-1, max=1)

        return out

    def compute_receptive_field(self):
        """Compute the receptive field in samples."""
        rf = self.hparams.kernel_size
        for n in range(1, self.hparams.nblocks):
            dilation = self.hparams.dilation_growth ** (n % self.hparams.stack_size)
            rf = rf + ((self.hparams.kernel_size - 1) * dilation)
        return rf

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- model related ---
        parser.add_argument("--ninputs", type=int, default=1)
        parser.add_argument("--noutputs", type=int, default=1)
        parser.add_argument("--nblocks", type=int, default=4)
        parser.add_argument("--kernel_size", type=int, default=5)
        parser.add_argument("--dilation_growth", type=int, default=10)
        parser.add_argument("--channel_growth", type=int, default=1)
        parser.add_argument("--channel_width", type=int, default=32)
        parser.add_argument("--stack_size", type=int, default=10)
        parser.add_argument("--grouped", default=False, action="store_true")
        parser.add_argument("--causal", default=False, action="store_true")
        parser.add_argument("--skip_connections", default=False, action="store_true")

        return parser


class TCNBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        dilation=1,
        cond_dim=2048,
        grouped=False,
        causal=False,
        conditional=False,
        **kwargs,
    ):
        super(TCNBlock, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.grouped = grouped
        self.causal = causal
        self.conditional = conditional

        groups = out_ch if grouped and (in_ch % out_ch == 0) else 1

        self.pad_length = (
            ((kernel_size - 1) * dilation)
            if self.causal
            else ((kernel_size - 1) * dilation) // 2
        )
        self.conv1 = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=self.pad_length,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        if grouped:
            self.conv1b = torch.nn.Conv1d(out_ch, out_ch, kernel_size=1)

        if conditional:
            self.film = FiLM(cond_dim, out_ch)
        self.bn = torch.nn.BatchNorm1d(out_ch)

        self.relu = torch.nn.LeakyReLU()
        self.res = torch.nn.Conv1d(
            in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False
        )

    def forward(self, x, p):
        x_in = x

        x = self.relu(self.bn(self.conv1(x)))
        x = self.film(x, p)

        x_res = self.res(x_in)

        if self.causal:
            x = x[..., : -self.pad_length]
        x += x_res

        return x


# 1-dimensional convolutional layer
# in the order of conv -> norm -> activation
class Conv1d_layer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        bias=True,
        norm="batch",
        activation="relu",
        mode="conv",
    ):
        super(Conv1d_layer, self).__init__()

        self.conv1d = nn.Sequential()

        """ padding """
        if mode == "deconv":
            padding = int(dilation * (kernel_size - 1) / 2)
            out_padding = 0 if stride == 1 else 1
        elif mode == "conv" or "alias_free" in mode:
            if padding == "SAME":
                pad = int((kernel_size - 1) * dilation)
                l_pad = int(pad // 2)
                r_pad = pad - l_pad
                padding_area = (l_pad, r_pad)
            elif padding == "VALID":
                padding_area = (0, 0)
            else:
                pass

        """ convolutional layer """
        if mode == "deconv":
            self.conv1d.add_module(
                "deconv1d",
                nn.ConvTranspose1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=out_padding,
                    dilation=dilation,
                    bias=bias,
                ),
            )
        elif mode == "conv":
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(
                f"{mode}1d",
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=0,
                    dilation=dilation,
                    bias=bias,
                ),
            )
        elif "alias_free" in mode:
            if "up" in mode:
                up_factor = stride * 2
                down_factor = 2
            elif "down" in mode:
                up_factor = 2
                down_factor = stride * 2
            else:
                raise ValueError("choose alias-free method : 'up' or 'down'")
            # procedure : conv -> upsample -> lrelu -> low-pass filter -> downsample
            # the torchaudio.transforms.Resample's default resampling_method is 'sinc_interpolation' which performs low-pass filter during the process
            # details at https://pytorch.org/audio/stable/transforms.html
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            self.conv1d.add_module(
                f"{mode}1d",
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=0,
                    dilation=dilation,
                    bias=bias,
                ),
            )
            self.conv1d.add_module(
                f"{mode}upsample",
                torchaudio.transforms.Resample(orig_freq=1, new_freq=up_factor),
            )
            self.conv1d.add_module(f"{mode}lrelu", nn.LeakyReLU())
            self.conv1d.add_module(
                f"{mode}downsample",
                torchaudio.transforms.Resample(orig_freq=down_factor, new_freq=1),
            )

        """ normalization """
        if norm == "batch":
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))
            # self.conv1d.add_module("batch_norm", nn.SyncBatchNorm(out_channels))

        """ activation """
        if "alias_free" not in mode:
            if activation == "relu":
                self.conv1d.add_module("relu", nn.ReLU())
            elif activation == "lrelu":
                self.conv1d.add_module("lrelu", nn.LeakyReLU())

    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv1d(input)
        return output


# Residual Block
# the input is added after the first convolutional layer, retaining its original channel size
# therefore, the second convolutional layer's output channel may differ
class Res_ConvBlock(nn.Module):
    def __init__(
        self,
        dimension,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        bias=True,
        norm="batch",
        activation="relu",
        last_activation="relu",
        mode="conv",
    ):
        super(Res_ConvBlock, self).__init__()

        if dimension == 1:
            self.conv1 = Conv1d_layer(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
                norm=norm,
                activation=activation,
            )
            self.conv2 = Conv1d_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                norm=norm,
                activation=last_activation,
                mode=mode,
            )
        elif dimension == 2:
            self.conv1 = Conv2d_layer(
                in_channels,
                in_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
                norm=norm,
                activation=activation,
            )
            self.conv2 = Conv2d_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
                norm=norm,
                activation=last_activation,
                mode=mode,
            )

    def forward(self, input):
        c1_out = self.conv1(input) + input
        c2_out = self.conv2(c1_out)
        return c2_out


# Convoluaionl Block
# consists of multiple (number of layer_num) convolutional layers
# only the final convoluational layer outputs the desired 'out_channels'
class ConvBlock(nn.Module):
    def __init__(
        self,
        dimension,
        layer_num,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="SAME",
        dilation=1,
        bias=True,
        norm="batch",
        activation="relu",
        last_activation="relu",
        mode="conv",
    ):
        super(ConvBlock, self).__init__()

        conv_block = []
        if dimension == 1:
            for i in range(layer_num - 1):
                conv_block.append(
                    Conv1d_layer(
                        in_channels,
                        in_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                        norm=norm,
                        activation=activation,
                    )
                )
            conv_block.append(
                Conv1d_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    norm=norm,
                    activation=last_activation,
                    mode=mode,
                )
            )
        elif dimension == 2:
            for i in range(layer_num - 1):
                conv_block.append(
                    Conv2d_layer(
                        in_channels,
                        in_channels,
                        kernel_size,
                        padding=padding,
                        dilation=dilation,
                        bias=bias,
                        norm=norm,
                        activation=activation,
                    )
                )
            conv_block.append(
                Conv2d_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                    norm=norm,
                    activation=last_activation,
                    mode=mode,
                )
            )
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.conv_block(input)


# Feature-wise Linear Modulation
class FiLM(nn.Module):
    def __init__(self, condition_len=2048, feature_len=1024):
        super(FiLM, self).__init__()
        self.film_fc = nn.Linear(condition_len, feature_len * 2)
        self.feat_len = feature_len

    def forward(self, feature, condition, sefa=None):
        # SeFA
        if sefa:
            weight = self.film_fc.weight.T
            weight = weight / torch.linalg.norm((weight + 1e-07), dim=0, keepdims=True)
            eigen_values, eigen_vectors = torch.eig(
                torch.matmul(weight, weight.T), eigenvectors=True
            )

            ####### custom parameters #######
            chosen_eig_idx = sefa[0]
            alpha = eigen_values[chosen_eig_idx][0] * sefa[1]
            #################################

            An = eigen_vectors[chosen_eig_idx].repeat(condition.shape[0], 1)
            alpha_An = alpha * An

            condition += alpha_An

        film_factor = self.film_fc(condition).unsqueeze(-1)
        r, b = torch.split(film_factor, self.feat_len, dim=1)
        return r * feature + b
