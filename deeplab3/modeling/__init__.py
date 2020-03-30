from deeplab3.modeling.deeplab import DeepLab
from deeplab3.modeling.late_fusion_hha import LateFusion, MidFusion
import os
import torch

def make_model(cfg, **kwargs):
    if cfg.MODEL.NAME == "deeplab":
        return DeepLab(cfg)
    elif cfg.MODEL.NAME == "latefusion":
        return LateFusion(cfg)
    elif cfg.MODEL.NAME == "midfusion":
        return MidFusion(cfg)
    else:
        raise ValueError("Model not implemented: {}".format(cfg.MODEL.NAME))


def load_model(cfg):
    model = make_model(cfg)

    if cfg.SYSTEM.CUDA:
        model = torch.nn.DataParallel(model, device_ids=cfg.SYSTEM.GPU_IDS)
        model = model.cuda()

    model_filepath = os.path.join(cfg.RESUME.DIRECTORY, cfg.RESUME.MODEL)
    if not os.path.isfile(model_filepath):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(model_filepath))

    checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))
    cfg.TRAIN.START_EPOCH = checkpoint['epoch']
    if cfg.SYSTEM.CUDA:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(model_filepath, checkpoint['epoch']))

    return model