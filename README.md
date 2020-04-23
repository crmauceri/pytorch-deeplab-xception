# pytorch-deeplab-xception

### Differences to parent repository
- [x] Load RGB-D datasets Cityscapes, COCO, SUNRGBD, and SceneNetRGBD
- [x] Optional RGB-D network input using 4th channel in first convolutional layer 
- [x] YACS configuration files

### Introduction
This is a PyTorch(1.4) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. 

### Installation
The code was tested with Anaconda and Python 3.8. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    Other dependencies:
    ```Shell
    conda install matplotlib pillow tqdm protobuf tabulate scipy numpy
    pip install tensorboardX yacs
    
    conda install -c conda-forge pycocotools
    ```
    
    For visualuzation scripts:
    ```Shell
    conda install pandas
    conda install -c conda-forge tensorflow 
    ```
    
2. Compile SceneNetRGBD protobuf files
   ```bash
   cd deeplab3/dataloaders/datasets
   make
   ```
    
3. Install as module:
   ```bash
   cd $root
   pip install -e .
   ```

### Model Zoo

#### RGB Models

From parent repo,

| Backbone  | Dataset  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet (pretrained)   | Pascal VOC 2012 | 78.43%     | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |

New models,

| Backbone  | Dataset  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet (pretrained)  | COCO        |      | [google drive]() |
| ResNet    | COCO         |      | [google drive]() |

#### RGBD Models

| Backbone  | Dataset  |mIoU in val |Pretrained Model|
| :-------- | :------------: |:---------: |:--------------:|
| ResNet    | COCO + Synthetic Depth |      | [google drive]() |
| ResNet    | Cityscapes           |      | [google drive]() |
| ResNet    | SceneNetRGBD         |      | [google drive]() |
   
### Training
Follow steps below to train your model:

0. Configuration files use [YACS](https://github.com/rbgirshick/yacs). Full list of defaults is in `deeplab3/config/defaults.py`. Configurations used to train model zoo are in `configs/`.  You must set `DATASET.ROOT` to match your dataset location, otherwise, you can change as many or as few parameters as you desire. 

1. Run training script as To train deeplabv3+ using COCO dataset and ResNet as backbone:
    ```bash
    python deeplab3/train.py <config_file> <optional_parameter_overrides>
    ```    
   For example, to train deeplabv3+ using SUNRGBD dataset and ResNet as backbone:
    ```bash
    python deeplab3/train.py configs/sunrgbd.yaml
    ```

### Visualization

Jupyter notebook `COCO_Data_Browser.ipynb` is provided for visualizing data and trained network results. 

Tensorboard can also be used to visualize loss and accuracy metrics during training.

```bash
tensorboard serve --log=run/<output_path>
```

### Acknowledgement
[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
