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
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    path = os.path.join(self.directory, run, 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        log_file.write(self.cfg.dump())
        log_file.close()