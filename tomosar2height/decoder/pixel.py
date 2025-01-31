import torch
import torch.nn as nn
import torch.nn.functional as F

from tomosar2height.block import ResnetBlockFC


class ConvDecoder(nn.Module):
    """
    Convolutional decoder with skip connections.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        leaky (bool): Whether to use leaky ReLU activation.
    """

    def __init__(self, in_channels=32, out_channels=1, leaky=False):
        super(ConvDecoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(288, out_channels, kernel_size=1)

        self.act = F.leaky_relu if leaky else F.relu

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        out = self.conv4(torch.cat([x, x1, x2, x3], dim=1))
        return out


class FCDecoder(nn.Module):
    """
    Fully-connected decoder with ResNet blocks.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_blocks (int): Number of ResNet blocks.
        leaky (bool): Whether to use leaky ReLU activation.
    """

    def __init__(self, in_channels=32, out_channels=1, n_blocks=5, leaky=False):
        super(FCDecoder, self).__init__()

        self.blocks = nn.ModuleList([
            ResnetBlockFC(in_channels) for _ in range(n_blocks)
        ])
        self.fc_out = nn.Linear(in_channels, out_channels)
        self.act = F.leaky_relu if leaky else F.relu

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.fc_out(self.act(x))
        return x


class PixelwiseDecoder(nn.Module):
    """
    Pixel-wise decoder for feature reconstruction.
    Args:
        hidden_dim (int): Number of hidden dimensions.
        out_dim (int): Number of output dimensions.
        output_size (int): Size of the output raster.
        leaky (bool): Whether to use leaky ReLU activation.
        sample_mode (str): Interpolation mode ('bilinear' or others).
        mode (str): Decoder mode ('conv' or 'fc').
        use_footprint (bool): Whether to use a footprint decoder.
    """

    def __init__(self, hidden_dim=32, out_dim=1, output_size=512, leaky=False,
                 sample_mode='bilinear', mode='conv', use_footprint=False, **kwargs):
        super().__init__()

        self.mode = mode
        self.use_footprint = use_footprint
        self.sample_mode = sample_mode
        self.output_size = output_size

        if mode == 'conv':
            self.conv_decoder = ConvDecoder(hidden_dim, out_dim, leaky)
            if use_footprint:
                self.conv_decoder_footprint = ConvDecoder(hidden_dim, out_dim)
        elif mode == 'fc':
            self.fc_decoder = FCDecoder(hidden_dim, out_dim, leaky)
            if use_footprint:
                self.fc_decoder_footprint = FCDecoder(hidden_dim, out_dim)
        else:
            raise ValueError("Invalid mode. Use 'conv' or 'fc'.")

    def forward(self, feature_planes):
        """
        Decode feature planes into raster outputs.
        Args:
            feature_planes (dict): Dictionary containing feature planes.
        Returns:
            Tuple of decoded outputs and optional footprint.
        """
        plane_type = list(feature_planes.keys())
        c = 0

        if 'xy' in plane_type:
            c += feature_planes['xy']
            c = F.interpolate(c, size=self.output_size, mode=self.sample_mode, align_corners=True)

        if 'image' in plane_type:
            c += F.interpolate(feature_planes['image'], size=self.output_size, mode=self.sample_mode,
                               align_corners=True)

        x_footprint = None
        if self.mode == 'conv':
            x = self.conv_decoder(c)
            x = x.permute(0, 2, 3, 1)
            if self.use_footprint:
                x_footprint = self.conv_decoder_footprint(c).permute(0, 2, 3, 1)
        else:
            c = c.permute(0, 2, 3, 1)
            x = self.fc_decoder(c)
            if self.use_footprint:
                x_footprint = self.fc_decoder_footprint(c)

        return x, x_footprint
