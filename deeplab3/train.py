import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from deeplab3.config.defaults import get_cfg_defaults
from deeplab3.dataloaders import make_data_loader
from deeplab3.modeling.sync_batchnorm.replicate import patch_replication_callback
from deeplab3.modeling import make_model
from deeplab3.utils.loss import SegmentationLosses
from deeplab3.utils.calculate_weights import calculate_weights_labels
from deeplab3.utils.lr_scheduler import LR_Scheduler
from deeplab3.utils.saver import Saver
from deeplab3.utils.summaries import TensorboardSummary
from deeplab3.utils.metrics import BatchEvaluator

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Define Saver
        self.saver = Saver(cfg)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': self.cfg.SYSTEM.NUM_WORKERS, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(cfg, **kwargs)

        assert(self.nclass == cfg.DATASET.N_CLASSES)

        # Define network
        self.model = make_model(cfg)

        if self.cfg.TRAIN.FINETUNE:
            #Trains backbone less aggressively than final layers
            train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.cfg.TRAIN.LR},
                            {'params': self.model.get_10x_lr_params(), 'lr': self.cfg.TRAIN.LR * 10}]
        else:
            train_params = self.model.parameters()

        # Define Optimizer
        self.optimizer = torch.optim.AdamW(train_params)

        # Define Criterion
        # whether to use class balanced weights
        if self.cfg.TRAIN.USE_BALANCED_WEIGHTS:
            classes_weights_path = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME+'_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weights_labels(cfg.DATASET.ROOT, cfg.DATASET.NAME, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=self.cfg.SYSTEM.CUDA).build_loss(mode=self.cfg.MODEL.LOSS_TYPE)

        # Define Evaluator
        self.evaluator = BatchEvaluator(self.nclass)
        # Define lr scheduler
        # self.scheduler = LR_Scheduler(self.cfg.TRAIN.LR_SCHEDULER, self.cfg.TRAIN.LR,
        #                                     self.cfg.TRAIN.EPOCHS, len(self.train_loader))

        # Using cuda
        if self.cfg.SYSTEM.CUDA:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.GPU_IDS)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.cfg.TRAIN.RESUME:
            model_filepath = os.path.join(self.cfg.RESUME.DIRECTORY, self.cfg.RESUME.MODEL)
            if not os.path.isfile(model_filepath):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(model_filepath))
            if cfg.SYSTEM.CUDA:
                checkpoint = torch.load(model_filepath, map_location=torch.device('cuda'))
            else:
                checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))
            self.cfg.TRAIN.START_EPOCH = checkpoint['epoch']
            if self.cfg.SYSTEM.CUDA:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not self.cfg.TRAIN.FINETUNE:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_filepath, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if self.cfg.TRAIN.FINETUNE:
            self.cfg.TRAIN.START_EPOCH  = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.cfg.SYSTEM.CUDA:
                image, target = image.cuda(), target.cuda()
            # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            try:
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
            except ValueError as e:
                print("{}: {}".format(e.message, sample['id']))

            # Show 10 * 3 inference results each epoch
           # if i % (num_img_tr // 10) == 0:
            #    global_step = i + num_img_tr * epoch
             #   self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.TRAIN.BATCH_SIZE + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.cfg.TRAIN.NO_VAL:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
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

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()[0]
        mIoU = self.evaluator.Mean_Intersection_over_Union()[0]
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.cfg.TRAIN.BATCH_SIZE + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
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
    trainer = Trainer(cfg)

    for epoch in range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.EPOCHS):
        trainer.training(epoch)
        if not cfg.TRAIN.NO_VAL and epoch % cfg.TRAIN.EVAL_INTERVAL == (cfg.TRAIN.EVAL_INTERVAL - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
