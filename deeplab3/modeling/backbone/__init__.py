from deeplab3.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, use_depth, use_deeplab_format=True):
    if use_depth:
        channels = 4
    else:
        channels = 3

    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=False, channels=channels, use_deeplab_format=use_deeplab_format)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm, pretrained=False, channels=channels)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm, pretrained=False, channels=channels)
    else:
        raise NotImplementedError
