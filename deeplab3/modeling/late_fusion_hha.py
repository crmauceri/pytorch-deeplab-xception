import torch
import torch.nn as nn
import torch.nn.functional as F
from deeplab3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from deeplab3.modeling.aspp import build_aspp
from deeplab3.modeling.decoder import build_decoder
from deeplab3.modeling.backbone import build_backbone


class MidFusion(nn.Module):

    def __init__(self, cfg):
        super(MidFusion, self).__init__()

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone_rgb = build_backbone(cfg, BatchNorm)
        self.backbone_hha = build_backbone(cfg, BatchNorm)

        self.aspp = build_aspp(cfg, BatchNorm)

        self.decoder = build_decoder(cfg, BatchNorm)

        if cfg.MODEL.FREEZE_BN:
            self.freeze_bn()

    def forward(self, input):
        x_rgb, low_level_feat_rgb = self.backbone_rgb(input[:, :3, :, :])
        x_hha, low_level_feat_hha = self.backbone_hha(input[:, 3:, :, :])

        x = torch.cat([x_rgb, x_hha], dim=1)
        low_level_feat = torch.cat([low_level_feat_rgb, low_level_feat_hha], dim=1)

        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone_rgb, self.backbone_hha]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class LateFusion(nn.Module):

    def __init__(self, cfg):
        super(LateFusion, self).__init__()

        if cfg.MODEL.SYNC_BN == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone_rgb = build_backbone(cfg, BatchNorm)
        self.backbone_hha = build_backbone(cfg, BatchNorm)

        self.aspp_rgb = build_aspp(cfg, BatchNorm)
        self.aspp_hha = build_aspp(cfg, BatchNorm)

        self.decoder = build_decoder(cfg, BatchNorm)

        if cfg.MODEL.FREEZE_BN:
            self.freeze_bn()

    def forward(self, input):
        x_rgb, low_level_feat_rgb = self.backbone_rgb(input[:, :3, :, :])
        x_hha, low_level_feat_hha = self.backbone_hha(input[:, 3:, :, :])

        x_rgb = self.aspp_rgb(x_rgb)
        x_hha = self.aspp_hha(x_hha)

        x = torch.cat([x_rgb, x_hha], dim=1)
        low_level_feat = torch.cat([low_level_feat_rgb, low_level_feat_hha], dim=1)

        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone_rgb, self.backbone_hha]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp_rgb, self.aspp_hha, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    from deeplab3.config.defaults import get_cfg_defaults
    from deeplab3.dataloaders.datasets.cityscapes import CityscapesSegmentation
    from torch.utils.data import DataLoader

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/HHA/cityscapes_hha_latefusion.yaml')
    cfg.merge_from_list(['MODEL.DECODER_DOUBLE', True, 'MODEL.ASPP_DOUBLE', True])

    model = MidFusion(cfg)
    model.eval()

    cityscapes_train = CityscapesSegmentation(cfg, split='val')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        output = model(sample['image'])
        break

    #input = (torch.rand(1, 3, 513, 513), torch.rand(1, 3, 513, 513))
    # output = model(input)
    # print(output.size())


