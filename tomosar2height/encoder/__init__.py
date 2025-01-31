from tomosar2height.encoder import unet, hourglass, pointnet

encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'hourglass': hourglass.HGFilter,
    'unet': unet.UNet,
}
