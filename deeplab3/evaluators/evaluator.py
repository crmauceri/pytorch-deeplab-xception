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
    # Input
    #   writer (TensorboardSummary)
    #   epoch (int) current epoch
    #   n_images (int) number of images in epoch
    def write_metrics(self, writer, epoch, n_images):
        raise NotImplementedError("Should be overwritten by child classes")

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