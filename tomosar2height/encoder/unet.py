import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups
    )


def upconv2x2(in_channels, out_channels, mode='transpose'):
    """2x2 upconvolution"""
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels)
        )


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1
    )


class DownConv(nn.Module):
    """
    Module performing two convolutions and one max-pooling operation.
    Each convolution is followed by a ReLU activation.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    Module performing two convolutions and one upconvolution.
    Each convolution is followed by a ReLU activation.
    """

    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        else:
            self.conv1 = conv3x3(self.out_channels, self.out_channels)

        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """Forward pass combining encoder and decoder pathways."""
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """
    U-Net: A convolutional encoder-decoder neural network based on https://arxiv.org/abs/1505.04597.

    Key Modifications:
    (1) Padding in 3x3 convolutions prevents loss of border pixels.
    (2) Merging outputs does not require cropping.
    (3) Residual connections can be used with `merge_mode='add'`.
    (4) Non-parametric upsampling reduces channel dimensionality using 1x1 convolution.
    """

    def __init__(self, num_classes, in_channels=3, depth=5, start_filts=64, up_mode='transpose', merge_mode='concat',
                 **kwargs):
        super(UNet, self).__init__()

        if up_mode not in ('transpose', 'upsample'):
            raise ValueError(f"Invalid up_mode: {up_mode}")

        if merge_mode not in ('concat', 'add'):
            raise ValueError(f"Invalid merge_mode: {merge_mode}")

        if up_mode == 'upsample' and merge_mode == 'add':
            raise ValueError("up_mode 'upsample' is incompatible with merge_mode 'add'.")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Encoder pathway
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = i < depth - 1

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # Decoder pathway
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # Encoder pathway
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # Decoder pathway
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        x = self.conv_final(x)
        return x

    # UNCOMMENT TO DEBUG FEATURE MAPS
    # def forward(self, x):
    #     encoder_outs = []
    #     feature_maps = []
    #
    #     # Encoder pathway, save outputs for merging and feature maps
    #     for i, module in enumerate(self.down_convs):
    #         layer_name = f"encoder_layer_{i}"
    #         x, before_pool = module(x)
    #         encoder_outs.append(before_pool)
    #         feature_maps.append((layer_name, before_pool))  # Store the feature map with its layer name
    #
    #     # Decoder pathway, save outputs for feature maps
    #     for i, module in enumerate(self.up_convs):
    #         layer_name = f"decoder_layer_{i}"
    #         before_pool = encoder_outs[-(i + 2)]
    #         x = module(before_pool, x)
    #         feature_maps.append((layer_name, x))  # Store the feature map with its layer name
    #
    #     # Final convolution layer
    #     final_layer_name = "final_layer"
    #     x = self.conv_final(x)
    #     feature_maps.append((final_layer_name, x))  # Store the final output as a feature map
    #
    #     return x, feature_maps


if __name__ == "__main__":
    # Testing the UNet implementation
    model = UNet(num_classes=1, depth=6, merge_mode='concat', in_channels=32, start_filts=32)
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    reso = 256
    x = np.zeros((1, 32, reso, reso))
    x[:, :, reso // 2 - 1, reso // 2 - 1] = np.nan
    x = torch.FloatTensor(x)

    out = model(x)
    nan_ratio = torch.sum(torch.isnan(out)).item() / (reso * reso)
    print(f"NaN ratio: {nan_ratio:.6f}")
