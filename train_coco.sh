#!/bin/bash
# Flag 1 turns on debugger
if [$1]; then
	CUDA_VISIBLE_DEVICES=1 python -m pdb train.py --backbone resnet --lr 0.01 --workers 4 --epochs 40 --batch-size 16 --checkname deeplab-resnet --eval-interval 1 --dataset coco
else
	CUDA_VISIBLE_DEVICES=1 python train.py --backbone resnet --lr 0.01 --workers 4 --epochs 80 --batch-size 16 --checkname deeplab-resnet --eval-interval 1 --dataset coco --resume run/coco/deeplab-resnet/experiment_10/checkpoint.pth.tar 
fi
