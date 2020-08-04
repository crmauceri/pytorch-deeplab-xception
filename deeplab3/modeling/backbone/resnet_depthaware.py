import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from deeplab3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from deeplab3.modeling.backbone.ops.depthconv.module import DepthConv
from deeplab3.modeling.backbone.ops.depthavgpooling.module import DepthAvgPooling

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = DepthConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.depth_downsample = nn.AvgPool2d(3, stride=stride, padding=dilation)
        self.stride = stride
        self.dilation = dilation

    def forward(self, input):
        x, depth = input

        print(depth.shape)
        print(x.shape)

        residual = x

        out = self.conv1(x, depth)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        depth = self.depth_downsample(depth)
        return (out, depth)

class DepthAwareResNet(nn.Module):

    def __init__(self, cfg, block, layers, BatchNorm=nn.BatchNorm2d):

        self.channels = 3
        self.inplanes = 64
        super(DepthAwareResNet, self).__init__()
        blocks = [1, 2, 4]
        if cfg.MODEL.OUT_STRIDE == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif cfg.MODEL.OUT_STRIDE == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = DepthConv(self.channels, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.downsample_depth_a = nn.AvgPool2d(7, padding=3, stride=2)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = DepthAvgPooling(kernel_size=3, stride=2, padding=1)
        self.downsample_depth_b = nn.AvgPool2d(3, padding=1, stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        current_stride = 4 #self.conv1.stride * self.maxpool.stride
        self._out_feature_strides = {"stem": current_stride}
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            self._out_feature_strides["res{}".format(i+2)] = current_stride = int(
                current_stride * np.prod([k.stride for k in layer])
            )

        self._out_feature_channels = {"stem": 64,
                                      "res2": 64 * block.expansion,
                                      "res3": 128 * block.expansion,
                                      "res4": 256 * block.expansion,
                                      "res5": 512 * block.expansion}

        self.use_deeplab_out = cfg.MODEL.RESNET.DEEPLAB_OUTPUT
        self._out_features = cfg.MODEL.RESNET.OUT_FEATURES

        if cfg.MODEL.BACKBONE_ZOO:
            self._load_pretrained_model(use_cuda=cfg.SYSTEM.CUDA)
        elif len(cfg.MODEL.PRETRAINED) > 0:
            self._load_pretrained_model(model_file=cfg.MODEL.PRETRAINED, use_cuda=cfg.SYSTEM.CUDA)
        else:
            print("Training backbone from scratch")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        outputs = {}

        img = input[:, :3, :, :].contiguous()
        depth = input[:, 3, :, :].unsqueeze(1).contiguous()

        x = self.conv1(img, depth=depth)
        x = self.bn1(x)
        x = self.relu(x)

        depth = self.downsample_depth_a(depth)
        x = self.maxpool(x, depth=depth)

        if "stem" in self._out_features:
            outputs['stem'] = x

        depth = self.downsample_depth_b(depth)
        print("Layer 1")
        x, depth = self.layer1((x, depth))
        if 'res2' in self._out_features:
            outputs['res2'] = x

        print("Layer 2")
        x, depth = self.layer2((x, depth))
        if 'res3' in self._out_features:
            outputs['res3'] = x

        print("Layer 3")
        x, depth = self.layer3((x, depth))
        if 'res4' in self._out_features:
            outputs['res4'] = x

        print("Layer 4")
        x, depth = self.layer4((x, depth))
        if 'res5' in self._out_features:
            outputs['res5'] = x

        if self.use_deeplab_out:
            return outputs['res5'], outputs['res2']
        return outputs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_file=None, zoo_url='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', use_cuda=True):
        if model_file:
            print("Loading pretrained ResNet model: {}".format(model_file))
            if use_cuda:
                pretrain_dict = torch.load(model_file, map_location=torch.device('cuda'))
            else:
                pretrain_dict = torch.load(model_file, map_location=torch.device('cpu'))
        else:
            print("Loading pretrained ResNet model: {}".format(zoo_url))
            pretrain_dict = model_zoo.load_url(zoo_url)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def DepthAwareResNet101(cfg, BatchNorm=nn.BatchNorm2d):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DepthAwareResNet(cfg, Bottleneck, [3, 4, 23, 3], BatchNorm=BatchNorm)
    return model

if __name__ == "__main__":
    import torch
    from deeplab3.config.defaults import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/coco_rgbd_finetune.yaml')

    model = DepthAwareResNet101(cfg, BatchNorm=nn.BatchNorm2d)
    input = torch.rand(1, 4, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
