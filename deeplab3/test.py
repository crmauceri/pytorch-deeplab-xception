from deeplab3.config.defaults import get_cfg_defaults

from deeplab3.utils.metrics import Evaluator
import numpy as np
import torch
import argparse
import os
from deeplab3.dataloaders import make_data_loader
from deeplab3.modeling.sync_batchnorm.replicate import patch_replication_callback
from deeplab3.utils.loss import SegmentationLosses
from deeplab3.utils.calculate_weights import calculate_weights_labels
from deeplab3.modeling.deeplab import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from tabulate import tabulate

class Tester:

    def __init__(self, cfg):
        self.cfg = cfg

        # Define Dataloader
        kwargs = {'num_workers': self.cfg.SYSTEM.NUM_WORKERS, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(cfg, **kwargs)

        # Define Model and Load from File
        self.model = self.load_model()

        # Define Criterion
        # whether to use class balanced weights
        if self.cfg.TRAIN.USE_BALANCED_WEIGHTS:
            classes_weights_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weights_labels(cfg.DATASET.ROOT, cfg.DATASET.NAME, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.cfg.SYSTEM.CUDA).build_loss(
            mode=self.cfg.MODEL.LOSS_TYPE)

        # Using cuda
        if self.cfg.SYSTEM.CUDA:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.GPU_IDS)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        self.evaluator = Evaluator(self.nclass)

    def load_model(self):
        model = DeepLab(self.cfg)

        if self.cfg.SYSTEM.CUDA:
            model = torch.nn.DataParallel(model, device_ids=self.cfg.SYSTEM.GPU_IDS)
            model = model.cuda()

        if not os.path.isfile(self.cfg.TRAIN.RESUME):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(self.cfg.TRAIN.RESUME))

        checkpoint = torch.load(self.cfg.TRAIN.RESUME, map_location=torch.device('cpu'))
        self.cfg.TRAIN.START_EPOCH = checkpoint['epoch']
        if self.cfg.SYSTEM.CUDA:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
        best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.cfg.TRAIN.RESUME, checkpoint['epoch']))
        return model

    def run(self, dataloader, class_filter=None):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')

        num_classes = dataloader.dataset.NUM_CLASSES

        test_loss = 0.0
        total_pix = np.zeros((num_classes,))
        total_photos = np.zeros((num_classes,))
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cfg.SYSTEM.CUDA:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

            # Calculate class frequency
            mask = (target>=0) & (target < num_classes)
            labels = target[mask].astype(np.uint8)
            total_pix += np.bincount(labels, minlength=num_classes)
            total_photos[np.unique(labels)] += 1


        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class, acc_class_tensor = self.evaluator.Pixel_Accuracy_Class()
        mIoU, mIOU_class_tensor = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        print('Results:')
        print('[numImages: %5d]' % (i * self.cfg.TRAIN.BATCH_SIZE + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        print('Class breakdown:')
        breakdown = {"Class": ['unknown'] + dataloader.dataset.class_names,
                 "N_Photos": total_photos,
                 "% Pixels": total_pix / total_pix.sum(),
                 "Accuracy": acc_class_tensor,
                 "mIoU": mIOU_class_tensor}

        print(tabulate(breakdown, headers="keys"))

        plt.figure()
        plt.imshow(self.evaluator.confusion_matrix)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Metric Calculation")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(cfg)

    torch.manual_seed(cfg.SYSTEM.SEED)
    trainer = Tester(cfg)
    trainer.run(trainer.val_loader)



if __name__ == "__main__":
   main()

# fine_args = get_cfg_defaults()
# fine_args.merge_from_file('configs/sunrgbd_finetune.yaml')
# fine_args.merge_from_list(["DATASET.USE_DEPTH", False,
#                           "TRAIN.RESUME", "run/sunrgbd/sunrgbd_rgbd_resnet_deeplab/experiment_2/checkpoint.pth.tar"])
#
# rgbd_args = get_cfg_defaults()
# rgbd_args.merge_from_file('configs/coco_rgbd.yaml')
# rgbd_args.merge_from_list(["DATASET.USE_DEPTH", False,
#                           "TRAIN.RESUME", "pretrained/deeplab-resnet-rgbd.pth"])
#
#
# rgbd_model = load_model(rgbd_args)
# fine_model = load_model(fine_args)
#
# rgbd_model.eval()
# fine_model.eval()