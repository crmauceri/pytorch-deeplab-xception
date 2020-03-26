from deeplab3.modeling.deeplab import DeepLab
from deeplab3.modeling.late_fusion_hha import LateFusion, MidFusion


def make_model(cfg, **kwargs):
    if cfg.MODEL.NAME == "deeplab":
        return DeepLab(cfg)
    elif cfg.MODEL.NAME == "latefusion":
        return LateFusion(cfg)
    elif cfg.MODEL.NAME == "midfusion":
        return MidFusion(cfg)
    else:
        raise ValueError("Model not implemented: {}".format(cfg.MODEL.NAME))