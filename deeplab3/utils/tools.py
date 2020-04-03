from deeplab3.dataloaders.utils import decode_segmap
import matplotlib.pyplot as plt
import numpy as np
import torch

def run_model(cfg, image, model):
    if cfg.SYSTEM.CUDA:
        image = image.cuda()
    with torch.no_grad():
        output = model(image)

    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)
    return pred

def display_grid(images, labels):
    n = len(labels)
    plt.figure(figsize=(16, 4))

    for ii in range(n):
        plt.subplot('1{}{}'.format(n, ii + 1))
        plt.imshow(images[ii])
        plt.title(labels[ii])
        plt.axis('off')

def display_prediction_grid(image, target, pred, dataset='cityscapes'):
    segmap = decode_segmap(target.numpy().squeeze(), dataset=dataset)
    segmap_pred = decode_segmap(pred.squeeze(), dataset=dataset)

    images = [image[:, :, :3], image[:, :, 3:].squeeze(), segmap, segmap_pred]
    labels = ['RGB', 'Depth', 'GT', 'Prediction']
    display_grid(images, labels)