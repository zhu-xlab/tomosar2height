import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from utils.coordinate import coordinate2index
from torch_scatter import scatter_mean


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """3x3 Convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    """2x2 Upsampling Convolution."""
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    """1x1 Convolution."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """Down Convolutional Block with optional pooling and communication."""

    def __init__(self, in_channels, out_channels, i, pooling, depth, sample_mode='bilinear'):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.downsample = i

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_comm = nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        )

        self.fc_c = nn.Linear(in_channels, out_channels)
        self.sample_mode = sample_mode
        self.depth = depth

        if i > 0:
            self.conv1x1 = conv1x1(in_channels, out_channels)

    @staticmethod
    def generate_plane_features(p, c, channel, reso_plane):
        """Generate plane features from scattered points."""
        xy = p.clone()[..., [0, 1]]
        index = coordinate2index(xy, reso_plane)

        fea_plane = c.new_zeros(p.size(0), channel, reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T

        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane)  # B x 512 x reso^2

        return fea_plane

    def sample_plane_feature(self, p, c):
        """Sample plane features."""
        xy = p.clone()[..., [0, 1]]
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        return F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)

    def forward(self, p, x, x_after_conv=None, c_last=None):
        x['xy'] = F.relu(self.conv1(x['xy']))
        x['xy'] = F.relu(self.conv2(x['xy']))  #

        channel = x['xy'].shape[1]
        reso_plane = x['xy'].shape[2]

        if x_after_conv == None:
            # i == 0
            x_after_conv = x_after_conv
        else:
            if self.downsample in np.arange(2, self.depth):
                # i == 2, 3, 4, 5 (depth - 1)
                x['xy'] = x['xy'] + self.conv1x1(self.pool(x_after_conv['xy']))

            else:
                # i == 1
                x['xy'] = x['xy'] + self.conv1x1(x_after_conv['xy'])

        x_after_conv = {}
        x_after_conv['xy'] = x['xy']

        c = 0

        c += self.sample_plane_feature(p, x['xy'])
        c = c.transpose(1, 2)
        c = self.fc_comm(c)

        if c_last == None:
            c = c
        else:
            c = c + self.fc_c(c_last)

        x['xy'] = self.generate_plane_features(p, c, channel, reso_plane)  # 1 x 32 x res x res

        before_pool = {}
        before_pool['xy'] = x['xy']

        if self.pooling:
            x['xy'] = self.pool(x['xy'])

        return x, before_pool, x_after_conv, c


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, i, depth,
                 merge_mode='concat', up_mode='transpose', sample_mode='bilinear'):

        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)
        if i == depth - 2:
            self.upconv_noup = conv1x1(self.in_channels, self.out_channels)

        self.in_channels = in_channels

        self.fc_comm = nn.Sequential(
            nn.Linear(out_channels, 2 * out_channels),
            nn.ReLU(),
            nn.Linear(2 * out_channels, out_channels)
        )

        self.fc_c = nn.Linear(in_channels, out_channels)

        if i == depth - 2:
            self.conv1x1 = conv1x1(in_channels, out_channels)
        else:
            self.conv1x1 = upconv2x2(in_channels, out_channels, mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        self.sample_mode = sample_mode
        self.depth = depth

    def generate_plane_features(self, p, c, channel, reso_plane):
        """Generate plane features using scatter operation."""
        xy = p.clone()[..., [0, 1]]
        index = coordinate2index(xy, reso_plane)

        fea_plane = c.new_zeros(p.size(0), channel, reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso x reso
        fea_plane = fea_plane.reshape(p.size(0), channel, reso_plane, reso_plane)  # B x 512 x reso x reso

        return fea_plane

    def sample_plane_feature(self, p, c):
        """Sample plane features."""
        xy = p.clone()[..., [0, 1]]
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def forward(self, p, from_down, from_up, x_after_conv, c_last, i):
        """ Forward pass

        Arguments:
            from_down: Tensor from the encoder pathway.
            from_up: UpConvolved tensor from the decoder pathway.
        """

        if i == self.depth - 2:
            from_up['xy'] = self.upconv_noup(from_up['xy'])
        else:
            from_up['xy'] = self.upconv(from_up['xy'])

        x = {}
        if self.merge_mode == 'concat':
            x['xy'] = torch.cat((from_up['xy'], from_down['xy']), 1)
        else:
            x['xy'] = from_up['xy'] + from_down['xy']

        x['xy'] = F.relu(self.conv1(x['xy']))
        x['xy'] = F.relu(self.conv2(x['xy']))

        channel = x['xy'].shape[1]

        reso_plane = x['xy'].shape[2]

        if x_after_conv == None:
            x_after_conv = x_after_conv
        else:
            x['xy'] = x['xy'] + self.conv1x1(x_after_conv['xy'])

        x_after_conv = {}
        x_after_conv['xy'] = x['xy']

        if i == self.depth - 2:
            return x, x_after_conv, c_last

        c = 0
        c += self.sample_plane_feature(p, x['xy'])
        c = c.transpose(1, 2)

        c = self.fc_comm(c)

        if c_last == None:
            c = c
        else:
            c = c + self.fc_c(c_last)

        x['xy'] = self.generate_plane_features(p, c, channel, reso_plane)  # [1, 32, res, res]

        return x, x_after_conv, c


class UNet(nn.Module):
    """
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=0,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat', **kwargs):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            if i == 0 or i == depth - 1:
                pooling = False
            else:
                pooling = True

            down_conv = DownConv(ins, outs, i, pooling, depth=depth)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, i, up_mode=up_mode,
                             merge_mode=merge_mode, depth=depth)
            self.up_convs.append(up_conv)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, p, x, c):
        encoder_outs = []
        x_after_conv = None

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool, x_after_conv, c = module(p, x, x_after_conv, c)
            encoder_outs.append(before_pool)
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x, x_after_conv, c = module(p, before_pool, x, x_after_conv, c, i)

        x = self.conv_final(x['xy'])

        return x

    # UNCOMMENT TO DEBUG FEATURE MAPS
    # def forward(self, p, x, c):
    #     encoder_outs = []
    #     x_after_conv = None
    #     feature_maps = []  # List to store feature maps
    #
    #     # Encoder pathway, save outputs for merging
    #     for i, module in enumerate(self.down_convs):
    #         x, before_pool, x_after_conv, c = module(p, x, x_after_conv, c)
    #         encoder_outs.append(before_pool)
    #         # Save feature map after each down convolution
    #         feature_maps.append((f'encoder_layer_{i}', before_pool))
    #
    #     # Decoder pathway
    #     for i, module in enumerate(self.up_convs):
    #         before_pool = encoder_outs[-(i + 2)]
    #         x, x_after_conv, c = module(p, before_pool, x, x_after_conv, c, i)
    #         # Save feature map after each up convolution
    #         feature_maps.append((f'decoder_layer_{i}', x))
    #
    #     # Final convolution layer
    #     x = self.conv_final(x['xy'])
    #     feature_maps.append(('final_layer', x))  # Save the final output feature map
    #
    #     return x, feature_maps
    #     """
    #     # Save the variable using torch.save
    #     torch.save(feature_maps, "featuremaps_alto.pkl")
    #     """


if __name__ == "__main__":
    model = UNet(1, depth=6, merge_mode='concat', in_channels=32, start_filts=32)
    print(model)
    print(sum(p.numel() for p in model.parameters()))

    reso = 256
    x = np.zeros((1, 32, reso, reso))
    x[:, :, int(reso / 2 - 1), int(reso / 2 - 1)] = np.nan
    x = {'xy': torch.FloatTensor(x)}
    p = torch.zeros([1, 100, 3], dtype=torch.float)
    c = torch.zeros([1, 100, 32], dtype=torch.float)  # (B, N, C)
    out = model(p, x, {'xy'}, c)
    print('%f' % (torch.sum(torch.isnan(out)).detach().cpu().numpy() / (reso * reso)))