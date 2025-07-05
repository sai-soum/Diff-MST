# Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
# Under MIT License
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm: bool = True):
        super(ConvBlock, self).__init__()
        self.use_batchnorm = use_batchnorm

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.use_batchnorm:
            init_bn(self.bn1)
            init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class Cnn14(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        sample_rate: float,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: float,
        fmax: float,
        use_batchnorm: bool = False,
        input_norm: str = "batchnorm",
    ):
        super(Cnn14, self).__init__()
        self.embed_dim = embed_dim
        self.use_batchnorm = use_batchnorm
        self.input_norm = input_norm

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(
            in_channels=1, out_channels=64, use_batchnorm=use_batchnorm
        )
        self.conv_block2 = ConvBlock(
            in_channels=64, out_channels=128, use_batchnorm=use_batchnorm
        )
        self.conv_block3 = ConvBlock(
            in_channels=128, out_channels=256, use_batchnorm=use_batchnorm
        )
        self.conv_block4 = ConvBlock(
            in_channels=256, out_channels=512, use_batchnorm=use_batchnorm
        )
        self.conv_block5 = ConvBlock(
            in_channels=512, out_channels=1024, use_batchnorm=use_batchnorm
        )
        self.conv_block6 = ConvBlock(
            in_channels=1024, out_channels=2048, use_batchnorm=use_batchnorm
        )

        self.fc_mid = nn.Linear(2048, embed_dim, bias=True)
        self.fc_side = nn.Linear(2048, embed_dim, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc_mid)
        init_layer(self.fc_side)

    def forward(self, x: torch.Tensor):
        """
        input (torch.Tensor): Waveform tensor with shape (batch_size, chs, seq_len)
        """
        batch_size, chs, seq_len = x.size()

        # compute mid and side signals
        if chs == 1:
            pass
        elif chs == 2:
            x_mid = (x[:, 0, :] + x[:, 1, :]) / 2
            x_side = (x[:, 0, :] - x[:, 1, :]) / 2
            # stack along batch dim
            x = torch.stack([x_mid, x_side], dim=1)
        else:
            raise ValueError(f"Invalid number of channels: {chs}")

        # move to batch dim
        x = x.view(batch_size * chs, seq_len)

        # extract logmel features
        x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        if self.input_norm == "batchnorm":
            # this normalizes over mel bins which is problematic for equalization
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
        elif self.input_norm == "minmax":
            x = x.clamp(-80, 40.0)  # clamp the logmels between -80 and 40
            x = (x + 80) / 120  # normalize the logmels between 0 and 1
            x = (x * 2) - 1  # normalize the logmels between -1 and 1
        elif self.input_norm == "none":
            pass
        else:
            raise ValueError(f"Invalid input_norm: {self.input_norm}")

        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        # move mid and side back to channel dim
        x = x.view(batch_size, chs, -1)

        if chs == 1:
            x_mid = x[:, 0, :]
            mid_embed = self.fc_mid(x_mid)
            side_embed = mid_embed
        elif chs == 2:
            x_mid = x[:, 0, :]
            x_side = x[:, 1, :]
            mid_embed = self.fc_mid(x_mid)
            side_embed = self.fc_side(x_side)

        return mid_embed, side_embed
