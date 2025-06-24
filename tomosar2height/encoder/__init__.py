from tomosar2height.encoder import unet, hourglass, pointnet, pointnetpp

encoder_dict = {
    'pointnet_local_pool': pointnet.LocalPoolPointnet,
    'pointnet_plus_plus': pointnetpp.PointNetPlusPlus,
    'hourglass': hourglass.HGFilter,
    'unet': unet.UNet,
}
