from typing import List

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_max

from utils.coordinate import coordinate2index
from tomosar2height.block import ResnetBlockFC
from tomosar2height.encoder.unet import UNet
from tomosar2height.encoder.alto import UNet as Alto


class LocalPoolPointnet(nn.Module):
    """
    PointNet-based encoder network with ResNet blocks for each point.
    Supports fixed number of input points and multiple plane types.

    Args:
        feature_dim (int): Output feature dimension.
        dim (int): Input points dimension.
        hidden_dim (int): Hidden dimension of the network.
        scatter_type (str): Feature aggregation type for local pooling ('max' or 'mean').
        unet_type (str): U-Net type ('unet' or 'alto').
        unet_kwargs (dict): U-Net parameters.
        plane_resolution (int): Resolution for plane feature generation.
        n_blocks (int): Number of ResNetBlockFC layers.
    """

    def __init__(
            self, feature_dim=128, dim=3, hidden_dim=128, scatter_type='max', unet_type='alto',
            unet_kwargs=None, plane_resolution=None, n_blocks=5
    ):
        super().__init__()
        self.c_dim = feature_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2 * hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, feature_dim)
        self.actvn = nn.ReLU()

        self.unet_type = unet_type
        if unet_type == 'unet':
            self.unet = UNet(feature_dim, in_channels=feature_dim, **unet_kwargs)
        elif unet_type == 'alto':
            self.unet = Alto(feature_dim, in_channels=feature_dim, **unet_kwargs)
        else:
            raise ValueError(f"Unknown unet_type: {unet_type}")

        self.reso_plane = plane_resolution

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError("Invalid scatter type")

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Input point cloud (B, N, 3), normalized to [0, 1].
        Returns:
            dict: Processed plane features.
        """
        coord, index = {}, {}

        coord['xy'] = inputs.clone()[:, :, [0, 1]]
        index['xy'] = coordinate2index(coord['xy'], self.reso_plane)

        net = self.fc_pos(inputs)
        net = self.blocks[0](net)
        fea = {}

        for block in self.blocks[1:]:
            pooled = self.pool_local(index['xy'], net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        net = self.actvn(net)
        net = self.fc_c(net)
        fea_plane = self.generate_plane_features(index, net, plane='xy')

        if self.unet_type == 'unet':
            fea['xy'] = self.unet(fea_plane)
        elif self.unet_type == 'alto':
            fea['xy'] = self.unet(inputs, {'xy': fea_plane}, net)

        return fea

    def pool_local(self, i, net):
        """Perform local pooling on the feature map."""
        bs, fea_dim = net.size(0), net.size(2)
        fea = self.scatter(net.permute(0, 2, 1), i, dim_size=self.reso_plane ** 2)
        if self.scatter == scatter_max:
            fea = fea[0]
        fea = fea.gather(dim=2, index=i.expand(-1, fea_dim, -1))
        return fea.permute(0, 2, 1)

    def generate_plane_features(self, index_dict, c, plane):
        """Generate plane features based on scatter operations."""
        index = index_dict.get(plane)
        if index is None:
            raise NotImplementedError(f"Plane type {plane} not implemented.")

        fea_plane = c.new_zeros(c.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)
        fea_plane = scatter_mean(c, index, out=fea_plane)

        return fea_plane.reshape(c.size(0), self.c_dim, self.reso_plane, self.reso_plane)


if __name__ == '__main__':
    coord = {'xy': torch.tensor([[[0., 0.], [0.3, 0.9], [0.9, 0.3], [0.9, 0.9], [0.1, 0.2]]])}
    reso_plane, c_dim = 2, 2
    index = {'xy': coordinate2index(coord['xy'], reso_plane)}
    c = coord['xy']
    fea_plane = c.new_zeros(c.size(0), c_dim, reso_plane ** 2)
    c = c.permute(0, 2, 1)
    fea_plane = scatter_mean(c, index['xy'], out=fea_plane)
    fea_plane = fea_plane.reshape(c.size(0), c_dim, reso_plane, reso_plane)
    print(fea_plane)
