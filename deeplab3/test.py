from deeplab3.config.defaults import get_cfg_defaults

from deeplab3.dataloaders.datasets.coco import COCOSegmentation
from deeplab3.dataloaders import custom_transforms as tr
from deeplab3.dataloaders.utils import decode_segmap
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from deeplab3.dataloaders import make_data_loader
from deeplab3.modeling.sync_batchnorm.replicate import patch_replication_callback
from deeplab3.modeling.deeplab import *


def load_model(args):
    model = DeepLab(args)

    if args.SYSTEM.CUDA:
        model = torch.nn.DataParallel(model, device_ids=args.SYSTEM.GPU_IDS)
        model = model.cuda()

    if not os.path.isfile(args.TRAIN.RESUME):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(args.TRAIN.RESUME))
    checkpoint = torch.load(args.TRAIN.RESUME, map_location=torch.device('cpu'))
    args.TRAIN.START_EPOCH = checkpoint['epoch']
    if args.SYSTEM.CUDA:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.TRAIN.RESUME, checkpoint['epoch']))
    return model

fine_args = get_cfg_defaults()
fine_args.merge_from_file('configs/sunrgbd_finetune.yaml')
fine_args.merge_from_list(["DATASET.USE_DEPTH", False,
                          "TRAIN.RESUME", "run/sunrgbd/sunrgbd_rgbd_resnet_deeplab/experiment_2/checkpoint.pth.tar"])

rgbd_args = get_cfg_defaults()
rgbd_args.merge_from_file('configs/coco_rgbd.yaml')
rgbd_args.merge_from_list(["DATASET.USE_DEPTH", False,
                          "TRAIN.RESUME", "pretrained/deeplab-resnet-rgbd.pth"])


rgbd_model = load_model(rgbd_args)
fine_model = load_model(fine_args)

rgbd_model.eval()
fine_model.eval()