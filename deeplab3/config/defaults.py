from yacs.config import CfgNode as CN
import torch

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.GPU_IDS = [0]
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4
# Disable CUDA
_C.SYSTEM.NO_CUDA = False
# Random Seed
_C.SYSTEM.SEED = 1

# Checkpoint variables
# Where to save stuff
_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESUME = False
_C.CHECKPOINT.DIRECTORY = ''
_C.CHECKPOINT.MODEL = 'checkpoint.pth.tar'
_C.CHECKPOINT.EXCLUDE = []

_C.TRAIN = CN()
# number of epochs to train (default: auto)
_C.TRAIN.EPOCHS = -1
# start epoch index
_C.TRAIN.START_EPOCH = 0
# batch size (default: auto)
_C.TRAIN.BATCH_SIZE = -1
# whether to use balanced weights
_C.TRAIN.USE_BALANCED_WEIGHTS = False
# Checkpoint file name
_C.TRAIN.CHECKNAME = 'deeplab-defaults'
# Finetuning on a different dataset
_C.TRAIN.FINETUNE = False
# Evaluation interval
_C.TRAIN.EVAL_INTERVAL = 1.0
# Skip validation during training
_C.TRAIN.NO_VAL = False
_C.TRAIN.VAL_MAX = -1

## Learning Optimizer Parameters
# Learning rate (default: auto)
_C.TRAIN.LR = 0.001

_C.TEST = CN()
# batch size (default: auto)
_C.TEST.BATCH_SIZE = -1
_C.TEST.MAX_ITER = -1

_C.MODEL = CN()
_C.MODEL.NAME = "deeplab"
# Backbone name : ['resnet', 'xception', 'drn', 'mobilenet']
_C.MODEL.BACKBONE = "resnet"
# Use backbone weights from model zoo
_C.MODEL.BACKBONE_ZOO = False
# Location of pretrained model for bootstraping backbone
_C.MODEL.PRETRAINED = ""
# Network output stride
_C.MODEL.OUT_STRIDE = 16
# Batch normalization sync (default: auto)
_C.MODEL.SYNC_BN = False
# Freeze batch normalization parameters
_C.MODEL.FREEZE_BN = False
# Loss function type : ['ce', 'focal']
_C.MODEL.LOSS_TYPE = 'ce'

# Late Fusion options
_C.MODEL.ASPP_DOUBLE = False
_C.MODEL.DECODER_DOUBLE = False

# Resnet specific variables
_C.MODEL.RESNET = CN()
# Output format, true is deeplab native, false is detectron2 compatible
_C.MODEL.RESNET.DEEPLAB_OUTPUT = True
# Which layer features to return from forward function
_C.MODEL.RESNET.OUT_FEATURES = ['res5']

_C.MODEL.MOBILENET = CN()
_C.MODEL.MOBILENET.WIDTH_MULT = 1.

_C.MODEL.VGG16 = CN()
_C.MODEL.VGG16.DEPTH_CONV = True
_C.MODEL.VGG16.BN = True

_C.MODEL.INPUT_CHANNELS = 4

_C.EVALUATOR = CN()
_C.EVALUATOR.NAME = "segmentation"

_C.DATASET = CN()
# ['pascal', 'coco', 'cityscapes', 'sunrgbd']
_C.DATASET.NAME = 'coco'
_C.DATASET.N_CLASSES = 81
# Root directory of dataset
_C.DATASET.ROOT = '../datasets/coco/'
# whether to use SBD dataset
_C.DATASET.USE_STD = True
# Base image size
_C.DATASET.BASE_SIZE = 513
# Crop image size
_C.DATASET.CROP_SIZE = 513
# Use RGB-D input
_C.DATASET.MODE = 'RGBD' #['RGBD', 'RGB', 'RGB_HHA']
_C.DATASET.SYNTHETIC = False
_C.DATASET.DARKEN = False

# Variables specific to coco loader
_C.DATASET.COCO = CN()
_C.DATASET.COCO.CATEGORIES = 'coco' #['coco', 'pascal', 'sunrgbd']

# Variables specific to cityscapes loader
_C.DATASET.CITYSCAPES = CN()
_C.DATASET.CITYSCAPES.GT_MODE = 'gtCoarse' #['gtCoarse', 'gtFine']
_C.DATASET.CITYSCAPES.TRAIN_SET = 'train_extra' #['train_extra', 'train']
_C.DATASET.CITYSCAPES.DEPTH_DIR = 'disparity' #['disparity', 'VNL_Monocular', 'HHA']

# Use Box-Cox Transform on Depth Data
_C.DATASET.POWER_TRANSFORM = False
# Box-Cox Lambda
_C.DATASET.PT_LAMBDA = -0.5

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  C = _C.clone()

  C.SYSTEM.CUDA = not C.SYSTEM.NO_CUDA and torch.cuda.is_available()

  if C.SYSTEM.CUDA and len(C.SYSTEM.GPU_IDS) > 1:
      C.MODEL.SYNC_BN = True
  else:
      C.MODEL.SYNC_BN = False

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

  # if C.TRAIN.LR == -1:
  #     lrs = {
  #         'coco': 0.1,
  #         'cityscapes': 0.01,
  #         'pascal': 0.007,
  #     }
  #     C.TRAIN.LR = lrs[C.DATASET.NAME.lower()] / (4 * len(C.SYSTEM.GPU_IDS)) * C.TRAIN.BATCH_SIZE

  if C.MODEL.BACKBONE == 'drn':
      C.MODEL.OUT_STRIDE = 8

  if C.MODEL.RESNET.DEEPLAB_OUTPUT:
      C.MODEL.RESNET.OUT_FEATURES = ['res5', 'res2']

  return C

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`