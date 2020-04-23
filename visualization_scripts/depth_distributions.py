import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

from deeplab3.config.defaults import get_cfg_defaults
from deeplab3.dataloaders.utils import sample_distribution

from deeplab3.dataloaders.datasets.cityscapes import CityscapesSegmentation
from deeplab3.dataloaders.datasets.coco import COCOSegmentation
from deeplab3.dataloaders.datasets.sunrgbd import RGBDSegmentation

def plot_distribution(a_list, bins):
    ax = plt.figure()
    bin_middle = np.array([(bins[x] + bins[x + 1]) / 2.0 for x in range(len(bins) - 1)])
    for a in a_list:
        n = np.histogram(a, bins=bins)[0]
        # print(sum(n))
        # y = scipy.stats.norm.pdf(b, 0, 1)
        plt.plot(bin_middle[np.nonzero(n)], n[np.nonzero(n)])

    return ax


sunrgbd_rgbd = get_cfg_defaults()
sunrgbd_rgbd.merge_from_file('configs/sunrgbd.yaml')

coco_synth = get_cfg_defaults()
coco_synth.merge_from_file('configs/coco_rgbd.yaml')

city_rgbd = get_cfg_defaults()
city_rgbd.merge_from_file('configs/cityscapes_rgbd.yaml')

city_synth = get_cfg_defaults()
city_synth.merge_from_file('configs/cityscapes_synthetic_rgbd.yaml')

city_synth_dist = sample_distribution(CityscapesSegmentation(city_synth, split='val'))
coco_synth_dist = sample_distribution(COCOSegmentation(coco_synth, split='val'))

city_rgbd_dist = sample_distribution(CityscapesSegmentation(city_rgbd, split='val'))
sunrgbd_rgbd_dist = sample_distribution(RGBDSegmentation(sunrgbd_rgbd, split='val'))

#city_synth_dist_val = sample_distribution(CityscapesSegmentation(city_synth, split='val'))
# city_synth_dist = sample_distribution(CityscapesSegmentation(city_synth, split='val'))
# coco_synth_dist = sample_distribution(COCOSegmentation(coco_synth, split='val'))


print('City VNL: {}'.format(scipy.stats.skew(city_synth_dist['samples'], axis=0)))
print('City: {}'.format(scipy.stats.skew(city_rgbd_dist['samples'], axis=0)))
print('SUNRGBD: {}'.format(scipy.stats.skew(sunrgbd_rgbd_dist['samples'], axis=0)))
print('COCO: {}'.format(scipy.stats.skew(coco_synth_dist['samples'], axis=0)))

city_depth = city_rgbd_dist['samples'][:, -1]
city_depth = city_depth[np.nonzero(city_depth)]

city_synth = city_synth_dist['samples'][:, -1]
city_synth= city_synth[np.nonzero(city_synth)]

coco_depth = coco_synth_dist['samples'][:, -1]
coco_depth = coco_depth[np.nonzero(coco_depth)]

sun_depth = sunrgbd_rgbd_dist['samples'][:, -1]
sun_depth = sun_depth[np.nonzero(sun_depth)]
sun_depth = sun_depth[sun_depth<255]

bins = np.linspace(-4,4, num=256)
plot_distribution([city_depth, sun_depth, city_synth, coco_depth], bins)
plt.legend(['Cityscapes', 'SUNRGB-D', 'Cityscapes VNL Synthetic Depth', 'COCO VNL Synthetic Depth'])
plt.show()

print('City: %f' % scipy.stats.skew(city_depth, axis=0))
print('SUNRGBD: %f' % scipy.stats.skew(sun_depth, axis=0))

print('City VNL: %f' % scipy.stats.skew(city_synth, axis=0))
print('COCO: %f' % scipy.stats.skew(coco_depth, axis=0))



#Normalized

city_color = city_rgbd_dist['samples'][:, :3]
city_color_norm = (city_color - city_color.mean())/city_color.std()
city_depth_norm = (city_depth - city_depth.mean())/city_depth.std()
city_synth_norm = (city_synth - city_synth.mean())/city_synth.std()
coco_depth_norm = (coco_depth - coco_depth.mean())/coco_depth.std()
sun_depth_norm = (sun_depth - sun_depth.mean())/sun_depth.std()

bins = np.linspace(city_color.min(), city_color.max(), 256)
plot_distribution([city_color[:,0], city_color[:,1], city_color[:,2]], bins)
plt.legend(['red', 'green', 'blue'])

bins = np.linspace(-2.2,4, num=256)
plot_distribution([city_depth_norm, sun_depth_norm, city_synth_norm, coco_depth_norm], bins)

# Number of zeros (holes in depth map)
# zero_val = (0 - city_depth.mean())/city_depth.std()
# city_zeros = np.nonzero(city_rgbd_dist['samples'][:, -1] == 0)[0].shape
# plt.plot(zero_val, city_zeros, 'xb')
#
# zero_val = (0 - sun_depth.mean())/sun_depth.std()
# sun_zeros = np.nonzero(sunrgbd_rgbd_dist['samples'][:, -1] == 0)[0].shape
# plt.plot(zero_val, sun_zeros, 'xy')

plt.legend(['Cityscapes', 'SUNRGB-D', 'Cityscapes VNL Synthetic Depth', 'COCO VNL Synthetic Depth'])
plt.show()

#Transformed
city_synth_transform, lmbda = scipy.stats.boxcox(city_synth)
city_synth_norm = (city_synth_transform - city_synth_transform.mean())/city_synth_transform.std()

print(scipy.stats.skew(city_synth_transform, axis=0))