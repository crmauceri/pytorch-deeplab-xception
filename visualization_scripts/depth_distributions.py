import matplotlib.pyplot as plt

from deeplab3.config.defaults import get_cfg_defaults
from deeplab3.dataloaders.utils import sample_distribution

from deeplab3.dataloaders.datasets.cityscapes import CityscapesSegmentation
from deeplab3.dataloaders.datasets.coco import COCOSegmentation
from deeplab3.dataloaders.datasets.sunrgbd import RGBDSegmentation

sunrgbd_rgbd = get_cfg_defaults()
sunrgbd_rgbd.merge_from_file('configs/sunrgbd.yaml')

coco_synth = get_cfg_defaults()
coco_synth.merge_from_file('configs/coco_rgbd.yaml')

city_rgbd = get_cfg_defaults()
city_rgbd.merge_from_file('configs/cityscapes_rgbd.yaml')

city_synth = get_cfg_defaults()
city_synth.merge_from_file('configs/cityscapes_synthetic_rgbd.yaml')

city_rgbd_dist = sample_distribution(CityscapesSegmentation(city_rgbd, split='val'))
#city_synth_dist_val = sample_distribution(CityscapesSegmentation(city_synth, split='val'))
city_synth_dist_test = sample_distribution(CityscapesSegmentation(city_synth, split='test'))
coco_synth_dist = sample_distribution(COCOSegmentation(coco_synth, split='val'))
sunrgbd_rgbd_dist = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='val'))


plt.hist(city_rgbd_dist['samples'][:, -1], bins='auto')
