import numpy as np
import os
import model_utils
import json
import traceback

def run_model(cfg, cfg_filepath, result_file, rerun=False):
    try:
        cfg.merge_from_list(['CHECKPOINT.DIRECTORY', os.path.dirname(cfg_filepath),
                             'TEST.MAX_ITER', 1000,
                             'MODEL.PRETRAINED', "",
                             # Since we're using saved models, pretrained weights will be overwritten anyway.
                             'SYSTEM.GPU_IDS', [0]])

        checkpoint_file = os.path.join(cfg.CHECKPOINT.DIRECTORY, 'checkpoint.pth.tar')
        if rerun or (os.path.exists(result_file) and (os.path.getmtime(result_file) > os.path.getmtime(checkpoint_file))):
            with open(result_file, 'r') as fp:
                metric_str = fp.read().split('{')[1].split('}')[0].replace("'", '"')
                metrics = json.loads('{' + metric_str + '}')
        else:
            metrics = model_utils.test_model(cfg, result_file)

        print("Success on {}: {:3.2f}".format(result_file, metrics['mIoU']))
        return True

    except Exception as e:
        print(e)
        traceback.print_exc()
        return False


def run_all_models(models, rerun=False):
    failed = []

    for cfg_filepath in models:
        cfg = model_utils.match_cfg_versions(cfg_filepath)
        result_file = os.path.join(cfg.CHECKPOINT.DIRECTORY, 'validation_report.txt')
        if not run_model(cfg, cfg_filepath, result_file, rerun):
            failed.append(cfg_filepath)

    print("Failed models: ".format("\n".join(failed)))


def run_low_light_models(low_light_models, gain, gamma, rerun=False):
    failed = []

    for cfg_filepath in low_light_models:
        for i in gain:
            for j in gamma:
                cfg = model_utils.match_cfg_versions(cfg_filepath)
                cfg.merge_from_list(['DATASET.DARKEN.DARKEN', True,
                                     'DATASET.DARKEN.GAIN', float(i),
                                     'DATASET.DARKEN.GAMMA', float(j)])

                result_file = os.path.join(cfg.CHECKPOINT.DIRECTORY,
                                           'validation_report_gain{:3.2f}_gamma{:3.2f}.txt'.format(float(i), float(j)))

                if not run_model(cfg, cfg_filepath, result_file, rerun):
                    failed.append(cfg_filepath)

    print("Failed models: ".format("\n".join(failed)))


if __name__ == "__main__":
    model_configs = model_utils.get_all_models("../run/cityscapes/")
    run_all_models(model_configs, True)

    model_configs = model_utils.get_all_models("../run/scenenet/")
    run_all_models(model_configs, True)

    model_configs = model_utils.get_all_models("../run/coco/")
    run_all_models(model_configs, True)

    # low_light_models = ['../run/older/cityscapes_rgbd_xception_fine/2020_04_26-20_04_07/parameters.txt.yaml',
    #                     '../run/older/cityscapes_rgb_xception_pt/2020_04_27-01_43_58/parameters.txt.yaml',
    #                     '../run/cityscapes/cityscapes_rgbd_xception_low_light/2020_09_25-19_32_43/parameters.yaml',
    #                     '../run/cityscapes/cityscapes_rgb_xception_low_light/2020_09_25-19_36_53/parameters.yaml',
    #                     '../run/scenenet/scenenet_rgbd_xception/2020_09_17-22_10_19/parameters.yaml',
    #                     '../run/scenenet/scenenet_rgb_xception/2020_09_17-22_14_43/parameters.yaml',
    #                     '../run/scenenet/scenenet_rgbd_xception_low_light/2020_09_25-23_11_51/parameters.yaml',
    #                     '../run/scenenet/scenenet_rgbd_xception_low_light/2020_09_28-08_36_05/parameters.yaml']
    #
    # gain = np.linspace(0.1, 1, 3).tolist()
    # gamma = np.linspace(1,3, 3).tolist()
    #
    # run_low_light_models(low_light_models, gain, gamma)
