import torch
import torch.nn as nn
from typing import Dict
from tomosar2height.decoder import decoder_dict
from tomosar2height.encoder import encoder_dict


class TomoSAR2Height(nn.Module):
    """ TomoSAR2Height network.

    Args:
        cfg: Configuration dictionary
    """

    def __init__(self, cfg):
        super().__init__()

        cfg_model = cfg['model']
        self.dim = cfg_model['data_dim']
        self.use_cloud = cfg.use_cloud
        self.use_image = cfg.use_image

        # Point cloud encoder
        if self.use_cloud:
            encoder_str = cfg_model['encoder']
            encoder_kwargs = cfg_model['encoder_kwargs']
            self.point_encoder = encoder_dict[encoder_str](dim=self.dim, **encoder_kwargs)

        # Image encoder
        if self.use_image:
            image_encoder_str = cfg_model.get('encoder2')
            image_encoder_kwargs = cfg_model.get('encoder2_kwargs', {})
            self.image_encoder = encoder_dict[image_encoder_str](**image_encoder_kwargs)

        # Decoder
        decoder_kwargs = cfg_model['decoder_pixel_kwargs']
        self.decoder = decoder_dict['pixel'](**decoder_kwargs)

        self.threshold = cfg['test']['threshold']
        z_bound = cfg['dataset']['normalize']['z_bound']
        self.z_scale = z_bound[1] - z_bound[0]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """ Initialize weights using Xavier initialization. """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_cloud=None, input_image=None):
        """ Performs a forward pass through the network.

        Args:
            input_cloud (tensor): Input point cloud
            input_image (tensor): Input images

        Returns:
            tuple: Decoded features (pa, pb)
        """
        assert self.use_image or self.use_cloud, "At least one input modality must be used."
        feature_planes = self.encode_inputs(input_cloud, input_image)
        pa, pb = self.decoder(feature_planes)
        return pa * self.z_scale, pb

    def encode_inputs(self, input_cloud=None, input_image=None):
        """ Encodes the input data.

        Args:
            input_cloud (tensor): Input point cloud
            input_image (tensor): Input images

        Returns:
            dict: Encoded feature planes
        """
        feature_planes = {}
        if self.use_cloud:
            cloud_features: Dict = self.point_encoder(input_cloud)
            feature_planes.update(cloud_features)
        if self.use_image:
            image_features: torch.Tensor = self.image_encoder(input_image)
            feature_planes['image'] = image_features
        return feature_planes
