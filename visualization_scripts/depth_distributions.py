import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from tqdm import tqdm

from PIL import Image

from deeplab3.config.defaults import get_cfg_defaults
from deeplab3.dataloaders.utils import sample_distribution

from deeplab3.dataloaders.datasets.cityscapes import CityscapesSegmentation
from deeplab3.dataloaders.datasets.coco import COCOSegmentation
from deeplab3.dataloaders.datasets.sunrgbd import RGBDSegmentation

from deeplab3.dataloaders.SampleLoader import SampleLoader

city_rgbd = get_cfg_defaults()
city_rgbd.merge_from_file('configs/cityscapes_rgbd.yaml')
city_rgbd.merge_from_list(['DATASET.ROOT', 'datasets/cityscapes/'])

sunrgbd_rgbd = get_cfg_defaults()
sunrgbd_rgbd.merge_from_file('configs/sunrgbd.yaml')
sunrgbd_rgbd.merge_from_list(['DATASET.ROOT', 'datasets/SUNRGBD/'])

sunrgbd_rgbd_dist_train = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='train'), n=100)
sunrgbd_rgbd_dist_test = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='test'), n=100)

city_rgbd_dist = sample_distribution(CityscapesSegmentation(city_rgbd, split='train'))
