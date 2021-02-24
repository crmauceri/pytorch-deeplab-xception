import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from tqdm import tqdm

from PIL import Image

from deeplab3.config.defaults import get_cfg_defaults
from dataloaders.utils import sample_distribution

from dataloaders.datasets.cityscapes import CityscapesSegmentation
from dataloaders.datasets.coco import COCOSegmentation
from dataloaders.datasets.sunrgbd import RGBDSegmentation

from dataloaders.SampleLoader import SampleLoader

city_rgbd = get_cfg_defaults()
city_rgbd.merge_from_file('configs/cityscapes_rgbd.yaml')
city_rgbd.merge_from_list(['DATASET.ROOT', 'datasets/cityscapes/',
                           'DATASET.CITYSCAPES.DEPTH_DIR', 'completed_depth'])

sunrgbd_rgbd = get_cfg_defaults()
sunrgbd_rgbd.merge_from_file('configs/sunrgbd.yaml')
sunrgbd_rgbd.merge_from_list(['DATASET.ROOT', 'datasets/SUNRGBD/'])

sunrgbd_rgbd_dist_train = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='train'), n=100)
sunrgbd_rgbd_dist_test = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='test'), n=100)

city_rgbd_dist = sample_distribution(CityscapesSegmentation(city_rgbd, split='train'))
