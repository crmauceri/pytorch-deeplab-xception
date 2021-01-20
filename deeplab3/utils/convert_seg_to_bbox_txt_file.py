from deeplab3.dataloaders import make_data_loader
from deeplab3.utils.model_utils import match_cfg_versions
import argparse, glob, os
import numpy as np
from tqdm import tqdm

def main(cfg):
    datasets = make_data_loader(cfg)
    for dataset in datasets[:3]:
        if dataset is not None:
            for ii, sample in enumerate(tqdm(dataset)):
                for jj in range(len(sample["id"])):
                    filepath = sample['id'][jj].replace('leftImg8bit', 'bbox').replace('png', 'txt')
                    dir = os.path.dirname(filepath)

                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    np.savetxt(filepath, sample['label'][jj], delimiter=",", fmt=['%d', '%10.8f', '%10.8f', '%10.8f', '%10.8f'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert instance segmentation annotation to yolo txt files")
    parser.add_argument('config_file', help='config file path')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg = match_cfg_versions(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_list(['DATASET.ANNOTATION_TYPE', 'bbox', \
                         'DATASET.NO_TRANSFORMS', True, \
                         'TRAIN.BATCH_SIZE', 1, \
                         'TEST.BATCH_SIZE', 1])
    print(cfg)

    main(cfg)