#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --gpu-ids 0 --checkname deeplab-resnet_sunspot --eval-interval 1 --dataset sunrgbd
