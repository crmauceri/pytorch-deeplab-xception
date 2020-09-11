import numpy as np
from collections import defaultdict
from deeplab3.evaluators.evaluator import Evaluator
import matplotlib.pyplot as plt

class SegmentationEvaluator(Evaluator):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    ##
    # Calculates and accumulates confusion matrix for new batch
    # Input
    #    gt_image (tensor) ground truth
    #    pre_image (tensor) prediction
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    ##
    # Reset confusion_matrix to zero. Call at start of every epoch
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    ##
    # Writes all the metrics to the tensorboard log file
    #   writer (TensorboardSummary)
    #   epoch (int) current epoch
    #   n_images (int) number of images in epoch
    def write_metrics(self, writer, epoch, n_images):
        Acc = self.Pixel_Accuracy()
        Acc_class = self.Pixel_Accuracy_Class()[0]
        mIoU = self.Mean_Intersection_over_Union()[0]
        FWIoU = self.Frequency_Weighted_Intersection_over_Union()

        writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        writer.add_scalar('val/mIoU', mIoU, epoch)
        writer.add_scalar('val/Acc', Acc, epoch)
        writer.add_scalar('val/Acc_class', Acc_class, epoch)
        writer.add_scalar('val/fwIoU', FWIoU, epoch)

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, n_images))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        return mIoU

    ##
    # Accumulate confusion matrix
    # Confusion matrix is num_class by num_class matrix that counts number of times
    # pixel with ground truth class x has been predicted as class y
    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    ##
    # Calculates per pixel accuracy ( true positives / total pixels ) from confusion matrix
    # If confusion matrix is None, uses class variable
    # Return
    #   accuracy
    def Pixel_Accuracy(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
        Acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
        return Acc

    ##
    # Calculates mean per pixel accuracy over classes from confusion matrix
    # If confusion matrix is None, uses class variable
    # Return
    #   accuracy
    #   mean class accuracy,
    #   tensor of each class
    def Pixel_Accuracy_Class(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
        Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
        return np.nanmean(Acc), Acc

    ##
    # Calculates mean intersection over union over classes from confusion matrix
    # If confusion matrix is None, uses class variable
    # Return
    #   mean class iou,
    #   tensor of each class
    def Mean_Intersection_over_Union(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
        MIoU = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))
        return np.nanmean(MIoU), MIoU

    ##
    # Calculates intersection over union weighted by class frequency from confusion matrix
    # If confusion matrix is None, uses class variable
    # Return
    #   mean class iou,
    #   tensor of each class
    def Frequency_Weighted_Intersection_over_Union(self, confusion_matrix=None):
        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
        freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
        iu = np.diag(confusion_matrix) / (
                    np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                    np.diag(confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


class ImageSegmentationEvaluator(SegmentationEvaluator):
    def __init__(self, num_class):
        super().__init__(num_class)
        self.images_by_accuracy = defaultdict(list)
        self.images_by_iou = defaultdict(list)
        self.image_stats = defaultdict(dict)

    def add_images(self, gt_image, pre_image, img_ids):
        for ii, img_id in enumerate(img_ids):
            confusion_matrix = self._generate_matrix(gt_image[ii, :, :], pre_image[ii, :, :])
            accuracy = self.Pixel_Accuracy(confusion_matrix)
            iou = self.Mean_Intersection_over_Union(confusion_matrix)[0]
            fw_iou = self.Frequency_Weighted_Intersection_over_Union(confusion_matrix)

            self.images_by_iou[iou].append(img_id)
            self.images_by_accuracy[accuracy].append(img_id)
            self.image_stats[img_id] = {'m_iou': iou,
                                        'accuracy': accuracy,
                                        'fw_iou': fw_iou}

    def bottom_n(self, n=10):
        return {'accuracy': {key: self.images_by_accuracy[key] for key in sorted(self.images_by_accuracy)[:n]},
                'm_iou': {key: self.images_by_iou[key] for key in sorted(self.images_by_iou)[:n]}}

    def top_n(self, n=10):
        return {'accuracy': {key: self.images_by_accuracy[key] for key in sorted(self.images_by_accuracy, reverse=True)[:n]},
                'm_iou': {key: self.images_by_iou[key] for key in sorted(self.images_by_iou, reverse=True)[:n]}}


if __name__ == "__main__":

    # Test accuracy calculations
    import torch

    img_eval = ImageSegmentationEvaluator(19)

    pred = np.ones((1, 100, 100), dtype='int64')
    gt = np.ones((1, 100, 100), dtype='int64')
    accuracy, iou = img_eval.add_images(gt, pred, 1)
    assert accuracy == 1.0
    assert iou == 1.0

    #IOU of class 1 is 2/4, IOU of class 0 is 0/2
    pred = np.ones((1, 2, 2), dtype='int64')
    gt = np.zeros((1, 2, 2), dtype='int64')
    gt[0:2, 0] = 1.0
    accuracy, iou = img_eval.add_images(gt, pred, 2)
    assert accuracy == 0.5
    assert iou == 0.25

    #IOU of class 1 is 1/3, IOU of class 2 is 1/3
    #Accuracy is 2/4
    gt = np.zeros((1, 2, 2), dtype='int64')
    gt[0:2, 0] = 1.0
    gt[0:2, 1] = 2.0
    pred = gt.transpose()

    accuracy, iou = img_eval.add_images(gt, pred, 3)
    assert accuracy == 0.5
    assert iou == 1.0/3.0

    print('Passed tests')

