import numpy as np
import os

from deeplab3.dataloaders import make_data_loader
from deeplab3.utils.model_utils import match_cfg_versions
from deeplab3.test import Tester

import json
import torch
import traceback

def test_model(cfg, report_file, confusion_file=None):
    torch.manual_seed(cfg.SYSTEM.SEED)
    train_loader, val_loader, test_loader, num_classes = make_data_loader(cfg)
    tester = Tester(cfg)
    output, mat, metrics = tester.run(val_loader, num_classes)

    with open(report_file, 'w') as f:
        f.write(output)

    if confusion_file is not None:
        sio.savemat(confusion_file, {'confusion': mat})

    return metrics

def run_model(cfg, cfg_filepath, result_file, rerun=False):
    try:
        cfg.merge_from_list(['CHECKPOINT.DIRECTORY', os.path.dirname(cfg_filepath),
                             'TEST.MAX_ITER', 1000,
                             'MODEL.PRETRAINED', "",
                             # Since we're using saved models, pretrained weights will be overwritten anyway.
                             'SYSTEM.GPU_IDS', [0]])

        checkpoint_file = os.path.join(cfg.CHECKPOINT.DIRECTORY, 'checkpoint.pth.tar')

        # If result_file doesnt exist, or manual rerun flag or model had been updated since the result_file was generated
        if not os.path.exists(result_file) or rerun or os.path.getmtime(result_file) < os.path.getmtime(checkpoint_file):
            metrics = test_model(cfg, result_file)
        else:
            with open(result_file, 'r') as fp:
                metric_str = fp.read().split('{')[1].split('}')[0].replace("'", '"')
                metrics = json.loads('{' + metric_str + '}')

        print("Success on {}: {:3.2f}".format(result_file, metrics['mIoU']))
        return True

    except Exception as e:
        print(e)
        traceback.print_exc()
        return False


def run_all_models(models, rerun=False):
    failed = []

    for cfg_filepath in models:
        try:
            cfg = match_cfg_versions(cfg_filepath)
            checkpoint_dir = os.path.dirname(cfg_filepath)
            result_file = os.path.join(checkpoint_dir, 'validation_report.txt')
            if not run_model(cfg, cfg_filepath, result_file, rerun):
                failed.append(cfg_filepath)
        except Exception as e:
            print(e)
            traceback.print_exc()
            failed.append(cfg_filepath)

    print("Failed models: ".format("\n".join(failed)))


def run_low_light_models(low_light_models, gain, gamma, rerun=False):
    failed = []

    for cfg_filepath in low_light_models:
        for i in gain:
            for j in gamma:
                try:
                    cfg = match_cfg_versions(cfg_filepath)
                    cfg.merge_from_list(['DATASET.DARKEN.DARKEN', True,
                                         'DATASET.DARKEN.GAIN', float(i),
                                         'DATASET.DARKEN.GAMMA', float(j)])

                    checkpoint_dir = os.path.dirname(cfg_filepath)
                    result_file = os.path.join(checkpoint_dir,
                                               'validation_report_gain{:3.2f}_gamma{:3.2f}.txt'.format(float(i), float(j)))

                    if not run_model(cfg, cfg_filepath, result_file, rerun):
                        failed.append(cfg_filepath)
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    failed.append(cfg_filepath)

    print("Failed models: ".format("\n".join(failed)))

def run_scrambled_models(models, rerun=False):
    failed = []

    for cfg_filepath in models:
        try:
            cfg = match_cfg_versions(cfg_filepath)
            cfg.merge_from_list(['DATASET.SCRAMBLED', True])

            checkpoint_dir = os.path.dirname(cfg_filepath)
            result_file = os.path.join(checkpoint_dir, 'validation_report_scrambled.txt')

            if not run_model(cfg, cfg_filepath, result_file, rerun):
                failed.append(cfg_filepath)
        except Exception as e:
            print(e)
            traceback.print_exc()
            failed.append(cfg_filepath)

    print("Failed models: ".format("\n".join(failed)))


if __name__ == "__main__":
    # model_configs = model_utils.get_all_models("../run/cityscapes/")
    # run_all_models(model_configs, True)
    #
    # model_configs = model_utils.get_all_models("../run/scenenet/")
    # run_all_models(model_configs, True)
    #
    # model_configs = model_utils.get_all_models("../run/coco/")
    # run_all_models(model_configs, True)

    low_light_models = ['../run/cityscapes/cityscapes_rgbd_xception_fine_coarse/2020_08_20-15_58_16/parameters.yaml',
                       '../run/cityscapes/cityscapes_rgb_xception_pt_fine_coarse/2020_08_03-15_41_22/parameters.yaml',
                        '../run/cityscapes/cityscapes_rgbd_xception_low_light/2020_09_25-19_32_43/parameters.yaml',
                        '../run/cityscapes/cityscapes_rgb_xception_low_light/2020_09_25-19_36_53/parameters.yaml',
                        '../run/scenenet/scenenet_rgbd_xception/2020_09_17-22_10_19/parameters.yaml',
                        '../run/scenenet/scenenet_rgb_xception/2020_09_17-14_43/parameters.yaml',
                        '../run/scenenet/scenenet_rgbd_xception_low_light/2020_09_25-23_11_51/parameters.yaml',
                        '../run/scenenet/scenenet_rgbd_xception_low_light/2020_09_28-08_36_05/parameters.yaml']
    #
    # gain = [0.33, 0.66, 1.0]
    # gamma = [1.0, 2.0, 3.0]
    # run_low_light_models(low_light_models, gain, gamma, True)

    run_scrambled_models(low_light_models, True)
