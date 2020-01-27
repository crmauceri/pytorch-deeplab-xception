from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.GPU_IDS = [0, 1]
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# Disable CUDA
_C.SYSTEM.NO_CUDA = False
# Random Seed
_C.SYSTEM.SEED = 1

_C.TRAIN = CN()
# number of epochs to train (default: auto)
_C.TRAIN.EPOCHS = -1
# start epoch index
_C.TRAIN.START_EPOCH = 0
# batch size (default: auto)
_C.TRAIN.BATCH_SIZE = -1
# whether to use balanced weights
_C.TRAIN.USE_BALANCED_WEIGHTS = False
# Resume training from checkpoint file
_C.TRAIN.RESUME = ""
# Checkpoint file name
_C.TRAIN.CHECKNAME = 'deeplab-resnet'
# Finetuning on a different dataset
_C.TRAIN.FINETUNE = False
# Evaluation interval
_C.TRAIN.EVAL_INTERVAL = 1
# Skip validation during training
_C.TRAIN.NO_VAL = False

## Learning Optimizer Parameters
# Learning rate (default: auto)
_C.TRAIN.LR = -1
# Learnign rate scheduler mode : ['poly', 'step', 'cos']
_C.TRAIN.LR_SCHEDULER = 'poly'
# Momentum
_C.TRAIN.MOMENTUM = 0.9
# Weight Decay
_C.TRAIN.WEIGHT_DECAY = 5e-4
# Whether to use nesterov
_C.TRAIN.NESTEROV = False


_C.TEST = CN()
# batch size (default: auto)
_C.TEST.BATCH_SIZE = -1

_C.MODEL = CN()
# Backbone name : ['resnet', 'xception', 'drn', 'mobilenet']
_C.MODEL.BACKBONE = "resnet"
# Network output stride
_C.MODEL.OUT_STRIDE = 16
# Batch normalization sync (default: auto)
_C.MODEL.SYNC_BN = False
# Freeze batch normalization parameters
_C.MODEL.FREEZE_BN = False
# Loss function type : ['ce', 'focal']
_C.MODEL.LOSS_TYPE = 'ce'


_C.DATASET = CN()
# ['pascal', 'coco', 'cityscapes', 'sunrgbd']
_C.DATASET.NAME = 'coco'
# Root directory of dataset
_C.DATASET.ROOT = '../datasets/coco/'
# whether to use SBD dataset
_C.DATASET.USE_STD = True
# Base image size
_C.DATASET.BASE_SIZE = 513
# Crop image size
_C.DATASET.CROP_SIZE = 513
# Use RGB-D input
_C.DATASET.USE_DEPTH = True

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  C = _C.clone()

  C.SYSTEM.CUDA = not C.SYSTEM.NO_CUDA and torch.cuda.is_available()

  # if C.MODEL.SYNC_BN is None:
  #     if C.SYSTEM.CUDA and len(C.SYSTEM.GPU_IDS) > 1:
  #         C.MODEL.SYNC_BN = True
  #     else:
  #         C.MODEL.SYNC_BN = False

  # default settings for epochs, batch_size and lr
  if C.TRAIN.EPOCHS == -1:
      epoches = {
          'coco': 30,
          'cityscapes': 200,
          'pascal': 50,
      }
      C.TRAIN.EPOCHS = epoches[C.DATASET.NAME.lower()]

  if C.TRAIN.BATCH_SIZE == -1:
      C.TRAIN.BATCH_SIZE = 4 * len(C.SYSTEM.GPU_IDS)

  if C.TEST.BATCH_SIZE == -1:
      C.TEST.BATCH_SIZE = C.TRAIN.BATCH_SIZE

  if C.TRAIN.LR == -1:
      lrs = {
          'coco': 0.1,
          'cityscapes': 0.01,
          'pascal': 0.007,
      }
      C.TRAIN.LR = lrs[C.DATASET.NAME.lower()] / (4 * len(C.SYSTEM.GPU_IDS)) * C.TRAIN.BATCH_SIZE

  return C

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`