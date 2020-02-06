from deeplab3.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(cfg, BatchNorm): #backbone, output_stride, BatchNorm, use_depth, use_deeplab_format=True):

    if cfg.MODEL.BACKBONE == 'resnet':
        return resnet.ResNet101(cfg, BatchNorm)
    elif cfg.MODEL.BACKBONE == 'xception':
        return xception.AlignedXception(cfg, BatchNorm)
    elif cfg.MODEL.BACKBONE == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif cfg.MODEL.BACKBONE == 'mobilenet':
        return mobilenet.MobileNetV2(cfg, BatchNorm)
    else:
        raise NotImplementedError
