import argparse
from math import log
from typing import Optional
import json
import os
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader

from retinanet import model
from retinanet.dataloader import CocoDataset, eval_collate, stack_labels
from retinanet.utils import get_logger, remove_module
from config import Config
from train import validate
from retinanet.coco_eval import evaluate_coco

assert torch.__version__.split(".")[0] == "1"

logger = get_logger(__name__)


def initialize_model(depth: int, num_classes):
    if depth == 18:
        retinanet = model.resnet18(num_classes=num_classes, pretrained=False,)
    elif depth == 34:
        retinanet = model.resnet34(num_classes=num_classes, pretrained=False,)
    elif depth == 50:
        retinanet = model.resnet50(num_classes=num_classes, pretrained=False,)
    elif depth == 101:
        retinanet = model.resnet101(num_classes=num_classes, pretrained=False,)
    elif depth == 152:
        retinanet = model.resnet152(num_classes=num_classes, pretrained=False,)
    else:
        raise ValueError(
            "Unsupported backbone specified, deppth must be one of 18, 34, 50, 101, 152"
        )
    return retinanet


def validate(model, dataset, valid_loader):
    model.eval()
    cls_loss, reg_loss = [], []
    val_image_ids, results = [], []

    for i, (images, labels, scales, offset_x, offset_y, image_ids) in tqdm(
        enumerate(valid_loader), total=len(valid_loader), leave=True
    ):

        val_image_ids.extend(image_ids)
        # logger.debug(Fore.YELLOW + f"batch id = {i}" + Style.RESET_ALL)
        # logger.debug(image_ids)

        with torch.no_grad():
            img_idx, confs, classes, bboxes, cl, reg = model(
                {"img": images.float().cuda(), "labels": stack_labels(labels).float().cuda(),}
            )
        img_idx = img_idx.cpu().numpy()
        confs = confs.cpu().numpy()
        classes = classes.cpu().numpy()
        bboxes = bboxes.cpu().numpy().astype(np.int32)
        cls_loss.append(cl.item())
        reg_loss.append(reg.item())

        if len(img_idx):
            # logger.debug(f"len(img_idx) = {len(img_idx)}")
            # logger.debug(f"img_idx = {img_idx}")

            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]

            for j, idx in enumerate(img_idx):
                imid = image_ids[idx]
                scale = scales[idx]
                ox, oy = offset_x[idx], offset_y[idx]
                score = confs[j]
                class_index = classes[j]
                bbox = bboxes[j]
                bbox[0] -= ox
                bbox[1] -= oy
                # bbox[2] -= ox
                # bbox[3] -= oy
                bbox = bbox / scale

                image_result = {
                    "image_id": imid,
                    "category_id": dataset.label_to_coco_label(class_index),
                    "score": float(score),
                    "bbox": bbox.tolist(),
                }
                results.append(image_result)

    return np.mean(cls_loss), np.mean(reg_loss), results, val_image_ids


def parse_args():
    parser = argparse.ArgumentParser("evaluate on test dataset")
    parser.add_argument("--image-dir", type=str, help="path to the test images")
    parser.add_argument("--json-path", type=str, help="path to coco json file")
    parser.add_argument("--checkpoint", type=str, help="path to saved checkpoint")
    parser.add_argument("--image-size", type=str, help="image size")
    parser.add_argument("--depth", type=int, help="model depth", default=50)
    parser.add_argument("--num-classes", type=int, help="number of classes in the model", default=1)
    parser.add_argument("--batch-size", type=int, help="batch size for eval", default=8)
    parser.add_argument(
        "--save-dir", type=str, help="directory to save the results", required=False
    )
    args = parser.parse_args()
    return args


def parse_image_size(img_size):
    img_size = [int(x) for x in img_size.split(",")]
    if len(img_size) < 2:
        img_size.append(img_size[0])
    return img_size


if __name__ == "__main__":
    # print(Path(__file__).parent)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_args()
    retinanet = initialize_model(args.depth, args.num_classes)
    retinanet.to(device)
    retinanet.freeze_bn()
    retinanet.eval()
    logger.info("successfully initialized model")

    state_dict = torch.load(args.checkpoint, map_location=device)
    state_dict = remove_module(state_dict)
    retinanet.load_state_dict(state_dict)
    logger.info(f"successfully loaded saved checkpoint")

    image_size = parse_image_size(args.image_size)
    dataset = CocoDataset(
        args.image_dir,
        args.json_path,
        image_size=image_size,
        normalize=Config.normalize,
        return_ids=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=Config.workers,
        collate_fn=eval_collate,
        pin_memory=True,
        drop_last=False,
    )
    _, _, results, val_image_ids = validate(retinanet, dataset, dataloader)
    temp_output_path = os.path.join(Path(__file__).parent, "eval_results.json")
    if len(results):
        with open(temp_output_path, "w") as f:
            json.dump(results, f, indent=4)

    stats = evaluate_coco(dataset, val_image_ids, predictions=temp_output_path)
    if args.save_dir is not None:
        map_avg, map_50, map_75, map_small = stats[:4]
        output_dict = {
            "map_avg": map_avg,
            "map_50": map_50,
            "map_75": map_75,
            "map_small": map_small,
        }
        with open(os.path.join(args.save_dir, "test_stats.json"), "w") as f:
            json.dump(output_dict, f, indent=2)

# print('CUDA available: {}'.format(torch.cuda.is_available()))


# def main(args=None):
#     parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

#     parser.add_argument('--coco_path', help='Path to COCO directory')
#     parser.add_argument('--model_path', help='Path to model', type=str)

#     parser = parser.parse_args(args)

#     dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
#                               transform=transforms.Compose([Normalizer(), Resizer()]))

#     # Create the model
#     retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

#     use_gpu = True

#     if use_gpu:
#         if torch.cuda.is_available():
#             retinanet = retinanet.cuda()

#     if torch.cuda.is_available():
#         retinanet.load_state_dict(torch.load(parser.model_path))
#         retinanet = torch.nn.DataParallel(retinanet).cuda()
#     else:
#         retinanet.load_state_dict(torch.load(parser.model_path))
#         retinanet = torch.nn.DataParallel(retinanet)

#     retinanet.training = False
#     retinanet.eval()
#     retinanet.module.freeze_bn()

#     coco_eval.evaluate_coco(dataset_val, retinanet)


# if __name__ == '__main__':
#     main()
