from deeplab3.config.defaults import get_cfg_defaults

from deeplab3.utils.metrics import BatchEvaluator, ImageEvaluator
import numpy as np
import scipy.io as sio
import torch
import argparse
import os
from deeplab3.dataloaders import make_data_loader
from deeplab3.modeling.sync_batchnorm.replicate import patch_replication_callback
from deeplab3.utils.loss import SegmentationLosses
from deeplab3.utils.calculate_weights import calculate_weights_labels
from deeplab3.modeling import load_model
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
        self.model = load_model(cfg)

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

        self.evaluator = BatchEvaluator(self.nclass)
        self.img_evaluator = ImageEvaluator()


    def run(self, dataloader, class_filter=None):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader, desc='\r')

        num_classes = dataloader.dataset.loader.NUM_CLASSES

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

        output = 'Results:\n'
        output += '[numImages: %5d]\n' % (i * self.cfg.TRAIN.BATCH_SIZE + image.data.shape[0])
        output += "Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}\n".format(Acc, Acc_class, mIoU, FWIoU)
        output += 'Loss: %.3f\n' % test_loss

        output += 'Class breakdown:\n'
        breakdown = {"Class": dataloader.dataset.loader.class_names,
                 "N_Photos": total_photos,
                 "% Pixels": total_pix / total_pix.sum(),
                 "Accuracy": acc_class_tensor,
                 "mIoU": mIOU_class_tensor}

        output += tabulate(breakdown, headers="keys")

        plt.figure()
        plt.imshow(self.evaluator.confusion_matrix)
        plt.show()

        print(output)

        return output, self.evaluator.confusion_matrix

    def rank_images(self, dataset):
        for i, sample in enumerate(dataset):
            image, target = sample['image'], sample['label']
            if self.cfg.SYSTEM.CUDA:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            self.img_evaluator.add_image(target, pred, i)

if __name__ == "__main__":
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
    output, mat = trainer.run(trainer.val_loader)

    with open(cfg.RESUME.DIRECTORY + 'report.txt', 'w') as f:
        f.write(output)

    sio.savemat(cfg.RESUME.DIRECTORY + 'confusion.mat', {'confusion': mat})