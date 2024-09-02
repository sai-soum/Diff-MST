# Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
# Under MIT License
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
# from torchlibrosa.augmentation import SpecAugmentation


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
    def __init__(self, in_channels, out_channels, use_batchnorm: bool = True, pool_type: str = 'avg'):
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

        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        if pool_type == "max":
            self.pool_fn = F.max_pool2d
        elif pool_type == "avg":
            self.pool_fn = F.avg_pool2d
        elif pool_type == "avg+max":
            def pool_avg_max(x: torch.Tensor, kernel_size: List[int]):
                return F.avg_pool2d(x, kernel_size) + F.max_pool2d(x, kernel_size)
            self.pool_fn = pool_avg_max
        else:
            raise Exception("Incorrect argument for `pool_type`!")
        

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.use_batchnorm:
            init_bn(self.bn1)
            init_bn(self.bn2)

    def forward(self, input: torch.Tensor, pool_size: List[int]):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = self.pool_fn(x, pool_size)

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
        num_classes: int,
        n_inputs: int = 1,
        use_batchnorm: bool = True,
    ):
        super(Cnn14, self).__init__()

        self.conv_block1 = ConvBlock(
            in_channels=n_inputs,
            out_channels=64,
            use_batchnorm=use_batchnorm,
        )
        self.conv_block2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            use_batchnorm=use_batchnorm,
        )
        self.conv_block3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            use_batchnorm=use_batchnorm,
        )
        self.conv_block4 = ConvBlock(
            in_channels=256,
            out_channels=512,
            use_batchnorm=use_batchnorm,
        )
        self.conv_block5 = ConvBlock(
            in_channels=512,
            out_channels=1024,
            use_batchnorm=use_batchnorm,
        )
        self.conv_block6 = ConvBlock(
            in_channels=1024,
            out_channels=2048,
            use_batchnorm=use_batchnorm,
        )

        self.fc = nn.Linear(2048, num_classes, bias=True)
        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn0)
        init_layer(self.fc)

    def forward(self, x: torch.Tensor):
        """
        input (torch.Tensor): Spectrogram tensor with shape (bs, chs, bins, frames)
        """
        batch_size, chs, bins, frames = x.size()

        # x = x.view(batch_size, -1)
        # x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        # x = x.transpose(1, 3)
        # x = self.bn0(x)
        # x = x.transpose(1, 3)
        # if self.training:
        #    x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 4))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(4, 2))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(4, 2))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(4, 2))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=2)  # mean across stft bins

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        # x = F.dropout(x, p=0.5, training=self.training)
        x_out = self.fc(x)
        clipwise_output = x_out

        return clipwise_output
