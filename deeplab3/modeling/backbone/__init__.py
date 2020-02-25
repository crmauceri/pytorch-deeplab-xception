from deeplab3.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(cfg, BatchNorm, name=None): #backbone, output_stride, BatchNorm, use_depth, use_deeplab_format=True):

    if name is None:
        name = cfg.MODEL.BACKBONE
    if name == 'resnet':
        return resnet.ResNet101(cfg, BatchNorm)
    elif name == 'xception':
        return xception.AlignedXception(cfg, BatchNorm)
    elif name == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif name == 'mobilenet':
        return mobilenet.MobileNetV2(cfg, BatchNorm)
    else:
        raise NotImplementedError
