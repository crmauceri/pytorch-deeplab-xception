import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    # Return
    #   mean class accuracy,
    #   tensor of each class
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        return np.nanmean(Acc), Acc

    # Return
    #   mean class iou,
    #   tensor of each class
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        return np.nanmean(MIoU), MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

class ImageEvaluator(object):
    def __init__(self, model):
        self.model = model
        self.images_by_accuracy = defaultdict(list)
        self.images_by_iou = defaultdict(list)
        self.image_stats = defaultdict(dict)

    def add_image(self, gt_image, pre_image, file_path):
        accuracy = np.sum(gt_image==pre_image)/np.numel(gt_image)
        intersection = np.sum(np.logical_and(gt_image, pre_image))
        union = np.sum(np.logical_and(gt_image, pre_image))+np.sum(np.logical_xor(gt_image, pre_image))
        iou = intersection/union

        self.images_by_iou[iou].append(file_path)
        self.images_by_accuracy[accuracy].append(file_path)
        self.image_stats[file_path] = {'iou': iou,
                                       'accuracy': accuracy}

    def top_n(self, n=10):
        return {'accuracy': {key: self.images_by_accuracy[key] for key in sorted(self.images_by_accuracy)[:n]},
                'iou': {key: self.images_by_iou[key] for key in sorted(self.images_by_iou)[:n]}}

    def bottom_n(self, n=10):
        return {'accuracy': {key: self.images_by_accuracy[key] for key in sorted(self.images_by_accuracy, reverse=True)[:n]},
                'iou': {key: self.images_by_iou[key] for key in sorted(self.images_by_iou, reverse=True)[:n]}}

    def display_image(self, file_path):
        image = plt.imread(file_path)
