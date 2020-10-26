# pytorch-retinanet

This repository is an extenstion of the original repository [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet).

## New features:
- ✅ Batched NMS for faster evaluation
- ✅ Automatic Mixed Precision (AMP) training
- ✅ Distributed training
    - ✅ DataParallel (DP)
    - ✅ Distributed Data Parallel
    - ✅ LARC (borrowed from `apex`)
- ✅ Augmentations
    - ✅ Flip
    - ✅ Rotate
    - ✅ Shear
    - ✅ Brightness
    - ✅ Contrast
    - ✅ Gamma
    - ✅ Saturation
    - ✅ Sharpen
    - ✅ Gaussian Blur
    - ⬜ RandAugment
- ✅ Cosine LR schedule with warmup
- ✅ Batch inference
- ✅ Export to ONNX
- ✅ Tensorboard logging
- ✅ Explicit negative sampling at batch level through `WeightedRandomSampler`
- ⬜ MLFlow tracking of experiments
- ⬜ Add docker

The above codebase is tested on:
- python 3.6, 3.7, 3.8
- CUDA 10.2
- CuDNN 8.0
- torch 1.6.0
- torchvision 0.7.0
- onnx 1.4.0, 1.5.1

## Usage

### 1. Single GPU
```bash
python train.py \
--dataset coco \
--image-dir <full-path-to-the-image-directory> \
--train-json-path <path-to-train-coco-json> \
--val-json-path <path-to-val-coco-json> \
--epochs 100 \
--depth 50 \
--batch-size 8 \
--num-workers 4 \
--base_lr 0.00001 \
--logdir <path-to-logging-directory>
```
If your dataset has images with empty annotations, you can choose to sample them in a certain raio at the batch level with the help of `--nsr` argument.

### 2. DataParallel (DP)
```bash
python train.py \
--dataset coco \
--image-dir <full-path-to-the-image-directory> \
--train-json-path <path-to-train-coco-json> \
--val-json-path <path-to-val-coco-json> \
--epochs 100 \
--depth 50 \
--batch-size 8 \
--num-workers 4 \
--logdir <path-to-logging-directory>
--dist-mode DP
```
You need to uncomment line 270 in `retinanet/model.py` in order for DP to work properly

### 3. DistributedDataParallel (DDP) 
Ex: Single Node with 4 GPUs.
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py \
--dataset coco \
--image-dir <full-path-to-the-image-directory> \
--train-json-path <path-to-train-coco-json> \
--val-json-path <path-to-val-coco-json> \
--epochs 100 \
--depth 50 \
--batch-size 8 \
--num-workers 4 \
--logdir <path-to-logging-directory>
```

For distributed training you need to `git checkout dist`.

## Run batch inference
```sh
python predict.py \
-i <path/to/test/images> \
-w <path/to/trained/checkpoint.pt> \
-o <path/to/output.json>
--batch-size 8 \
--num-workers 4 \
--num-class 1
--confidence 0.5
```

## Export to ONNX
```sh
python export.py \
--checkpoint <path/to/trained/checkpoint.pt>
--output-path <path/to/output.onnx>
```
