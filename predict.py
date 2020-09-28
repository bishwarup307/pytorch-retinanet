"""
__author__: bishwarup
created: Monday, 28th September 2020 11:06:26 pm
"""

import os
import json
import numpy as np
from tqdm import tqdm
import glob
import argparse
import torch
from torch.utils.data import DataLoader

from retinanet.dataloader import ImageDirectory, custom_collate
from retinanet.utils import get_logger
from retinanet import model

logger = get_logger(__name__)
valid_backbones = ("resnet-18", "resnet-34", "resnet-50", "resnet-101", "resnet-152")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict with retinanet model")
    parser.add_argument(
        "-i", "--image-dir", type=str, help="path to directory containing inference images"
    )
    parser.add_argument("-w", "--weights", type=str, help="path to saved checkpoint")
    parser.add_argument("-o", "--output", type=str, help="path to output")
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size for inference")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="number of multiprocessing workers"
    )
    parser.add_argument("--backbone", type=str, default="resnet-50", help="backbone model arch")
    parser.add_argument("--num-class", type=int, help="number of classes for the model")
    parser.add_argument(
        "--confidence", type=float, default=0.6, help="minimum confidence for the predictions"
    )
    parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")
    opt = parser.parse_args()

    if opt.backbone not in valid_backbones:
        raise AttributeError(f"unknown backbone. we only support {', '.join(valid_backbones)}")

    if not opt.output.endswith(".json"):
        raise AttributeError(f"output must be a path to `json` file, got {opt.output}")

    dataset = ImageDirectory(opt.image_dir)
    logger.info(f"running inference on {len(dataset)} images (batch-size = {opt.batch_size}).")
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        collate_fn=custom_collate,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if opt.backbone == "resnet-18":
        model = model.resnet18(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-34":
        model = model.resnet34(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-50":
        model = model.resnet50(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-101":
        model = model.resnet101(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-152":
        model = model.resnet152(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    else:
        raise NotImplementedError

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"using device {device}")

    ckpt = torch.load(opt.weights, map_location=device)
    model.load_state_dict(ckpt.state_dict())
    model.to(device)
    model.eval()
    logger.info(f"successfully loaded saved checkpoint.")

    preds = dict()
    for i, (batch, filenames) in tqdm(enumerate(loader), total=len(loader)):
        with torch.no_grad():
            img_id, confs, classes, bboxes = model(batch[0].float().cuda())
        img_id = img_id.cpu().numpy().tolist()
        confs = confs.cpu().numpy()
        classes = classes.cpu().numpy()
        bboxes = bboxes.cpu().numpy().astype(np.int32)

        for i, imgid in enumerate(img_id):
            f = filenames[imgid]
            pr = {
                "bbox": bboxes[i].tolist(),
                "confidence": float(confs[i]),
                "class_index": int(classes[i]),
            }
            if f in preds:
                preds[f].append(pr)
            else:
                preds[f] = [pr]

    with open(opt.output, "w") as f:
        json.dump(preds, f, indent=2)
    logger.info(f"predictions saved in {opt.output}")
