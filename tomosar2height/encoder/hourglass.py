import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias
    )


class Flatten(nn.Module):
    """Flatten layer for reshaping the tensor to a 2D array"""

    def forward(self, input):
        return input.view(input.size(0), -1)


class ConvBlock(nn.Module):
    """
    Convolutional block with three convolutional layers and normalization.
    Supports batch and group normalization.
    """

    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=1,
                    bias=False
                ),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = F.relu(self.bn1(x), inplace=True)
        out1 = self.conv1(out1)

        out2 = F.relu(self.bn2(out1), inplace=True)
        out2 = self.conv2(out2)

        out3 = F.relu(self.bn3(out2), inplace=True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    """
    HourGlass module for feature extraction at multiple scales.
    Supports recursive structure.
    """

    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))
        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = self._modules['b1_' + str(level)](inp)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        low3 = self._modules['b3_' + str(level)](low2)

        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    """
    HourGlass-based feature extraction network.
    """

    def __init__(self, in_channel, feature_dim=256, num_hourglass=2, num_stack=4, norm='group', hg_down='ave_pool'):
        super(HGFilter, self).__init__()

        self.in_channel = in_channel
        self.out_feature_dim = feature_dim
        self.num_hourglass = num_hourglass
        self.num_modules = num_stack
        self.norm = norm
        self.hg_down = hg_down

        # Base part
        self.conv1 = nn.Conv2d(self.in_channel, 64, kernel_size=7, stride=2, padding=3)

        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.norm)
            self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.norm)
            self.down_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        elif self.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.norm)
        else:
            raise NameError('Unknown HGFilter downsampling method!')

        self.conv3 = ConvBlock(128, 128, self.norm)
        self.conv4 = ConvBlock(128, 256, self.norm)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, self.num_hourglass, 256, self.norm))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256, self.norm))
            self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))

            if self.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(32, 256))

            self.add_module('l' + str(hg_module),
                            nn.Conv2d(256, self.out_feature_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module),
                                nn.Conv2d(self.out_feature_dim, 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown HGFilter downsampling method!')

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x
        outputs = []

        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)
            ll = self._modules['top_m_' + str(i)](hg)
            ll = F.relu(self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll)), inplace=True)
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs[-1]


if __name__ == '__main__':
    opt = {
        'in_channel': 2,
        'num_hourglass': 2,
        'feature_dim': 256,
        'num_stack': 4,
        'norm': 'group',
        'hg_down': 'ave_pool'
    }

    model = HGFilter(**opt)
    print(model)
