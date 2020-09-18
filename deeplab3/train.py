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
from deeplab3.evaluators import make_evaluator

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
            train_params = [{'params':self.model.parameters(), 'lr': self.cfg.TRAIN.LR}]

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
        self.evaluator = make_evaluator(self.cfg, self.nclass)

        # Using cuda
        if self.cfg.SYSTEM.CUDA:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.SYSTEM.GPU_IDS)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if self.cfg.CHECKPOINT.RESUME or self.cfg.TRAIN.FINETUNE:
            model_filepath = os.path.join(self.cfg.CHECKPOINT.DIRECTORY, self.cfg.CHECKPOINT.MODEL)

            if not os.path.isfile(model_filepath):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(model_filepath))
            if cfg.SYSTEM.CUDA:
                checkpoint = torch.load(model_filepath, map_location=torch.device('cuda'))
            else:
                checkpoint = torch.load(model_filepath, map_location=torch.device('cpu'))
            self.cfg.TRAIN.START_EPOCH = checkpoint['epoch']

            strict = True
            for layer in cfg.CHECKPOINT.EXCLUDE:
                del checkpoint['state_dict'][layer]
                strict = False
            if self.cfg.SYSTEM.CUDA:
                self.model.module.load_state_dict(checkpoint['state_dict'], strict=strict)
            else:
                self.model.load_state_dict(checkpoint['state_dict'], strict=strict)

            # Load optimizer parameters and best previous prediction if resuming
            if self.cfg.CHECKPOINT.RESUME:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_pred = checkpoint['best_pred']

            # Clear start epoch if fine-tuning
            if self.cfg.TRAIN.FINETUNE:
                self.cfg.TRAIN.START_EPOCH = 0

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_filepath, checkpoint['epoch']))

        # What is the accuracy before training?
        self.validation(0)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        val_interval = math.floor(num_img_tr * self.cfg.TRAIN.EVAL_INTERVAL)
        for i, sample in enumerate(tbar):
            iter = i + num_img_tr * epoch
            if sample is not None:
                image, target = sample['image'], sample['label']
                if self.cfg.SYSTEM.CUDA:
                    image, target = image.cuda(), target.cuda()

                self.optimizer.zero_grad()
                try:
                    output = self.model(image)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
                    self.writer.add_scalar('train/total_loss_iter', loss.item(), iter)
                except ValueError as e:
                    print("{}: {}".format(str(e), sample['id']))

            if mod(i+1, val_interval) == 0:
                if not cfg.TRAIN.NO_VAL:
                    trainer.validation(epoch, iter)

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


    def validation(self, epoch, iter):
        self.model.eval()
        self.evaluator.reset()

        max_val = min(self.cfg.TRAIN.VAL_MAX, len(self.val_loader))
        if max_val == -1:
            max_val = len(self.val_loader)
        tbar = tqdm(self.val_loader, desc='\r', total=max_val)
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

            if i==max_val:
                break

        # Fast test during the training
        new_pred = self.evaluator.write_metrics(self.writer, iter, i * self.cfg.TRAIN.BATCH_SIZE + image.data.shape[0])

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

    trainer.writer.close()

if __name__ == "__main__":
   main()
