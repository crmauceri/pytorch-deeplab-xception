import os, datetime
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.directory = os.path.join('run', cfg.DATASET.NAME, cfg.TRAIN.CHECKNAME)
        self.runs = sorted(glob.glob(os.path.join(self.directory, '*')))
        if not self.cfg.CHECKPOINT.RESUME:
            x = datetime.datetime.now()
            run_id = x.strftime("%Y_%m_%d-%H_%M_%S")
            self.experiment_dir = os.path.join(self.directory, run_id)
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
        else:
            self.experiment_dir = cfg.CHECKPOINT.DIRECTORY

        print("Saver configured to save checkpoints to: " + self.experiment_dir)

    def save_checkpoint(self, state, is_best):
        """Saves checkpoint to disk"""
        filename = self.cfg.CHECKPOINT.MODEL
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.yaml')
        log_file = open(logfile, 'w')
        log_file.write(self.cfg.dump())
        log_file.close()


def find_best_checkpoint(parent_dir):
    max_miou = 0.0
    filepath = None
    for run in os.path.dir(parent_dir):
        path = os.path.join(parent_dir, run, 'best_pred.txt')
        if os.path.exists(path):
            with open(path, 'r') as f:
                miou = float(f.readline())
                if miou > max_miou:
                    max_miou = miou
                    filepath = run
    if filepath is not None:
        return os.path.join(parent_dir, filepath, 'checkpoint.pth.tar')
    else:
        return None