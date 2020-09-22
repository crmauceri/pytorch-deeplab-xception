import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class Evaluator(object):
    ##
    # Constructor
    # Input
    #    num_class (int) number of classes to evaluate
    def __init__(self, num_class):
        self.num_class = num_class

    ##
    # Writes all the metrics to the tensorboard log file
    #   writer (TensorboardSummary)
    #   iter (int) current number of iterations completed
    #   n_images (int) number of images in epoch
    def write_metrics(self, writer, iter, n_images):
        output, _temp = self.calc_metrics()

        for metric, value in output.items():
            writer.add_scalar('val/{}'.format(metric), value, iter)

        print('Validation:')
        print('[Iter: %d, numImages: %5d]' % (iter, n_images))
        print(output)

        return output

    ##
    # Calculates all the metrics for evaluation
    # Return
    #   summary_metrics - dictionary of metrics with name, value pairs
    #   per_class_metrics - dictionary of metric tensors with name, tensor pairs
    def calc_metrics(self):
        raise NotImplementedError('Needs to be implemented by child class')

    ##
    # Calculates and accumulates evaluation metrics for new batch
    # Input
    #    gt_image (tensor) ground truth
    #    pre_image (tensor) prediction
    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape

    ##
    # Reset any member variables at the start of a new epoch
    def reset(self):
        pass