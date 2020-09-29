from pycocotools.cocoeval import COCOeval
import json
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from colorama import Fore, Style

from retinanet.dataloader import eval_collate
from retinanet.utils import get_logger

logger = get_logger(__name__, level="info")


def evaluate_coco(dataset, model, logdir, batch_size, num_workers, threshold=0.05):

    model.eval()
    results = []
    val_image_ids = []

    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=eval_collate,
        pin_memory=True,
        drop_last=False,
    )

    for i, (images, labels, scales, image_ids) in tqdm(
        enumerate(valid_loader), total=len(valid_loader)
    ):
        val_image_ids.extend(image_ids)
        logger.debug(Fore.YELLOW + f"batch id = {i}" + Style.RESET_ALL)
        logger.debug(image_ids)

        with torch.no_grad():
            img_idx, confs, classes, bboxes = model(images.float().cuda())
        img_idx = img_idx.cpu().numpy()
        confs = confs.cpu().numpy()
        classes = classes.cpu().numpy()
        bboxes = bboxes.cpu().numpy().astype(np.int32)

        if len(img_idx):

            logger.debug(f"len(img_idx) = {len(img_idx)}")
            # logger.debug(f"img_idx = {img_idx}")

            bboxes[:, 2] -= bboxes[:, 0]
            bboxes[:, 3] -= bboxes[:, 1]

            for j, idx in enumerate(img_idx):
                imid = image_ids[idx]
                scale = scales[idx]
                score = confs[j]
                class_index = classes[j]
                bbox = bboxes[j] / scale

                image_result = {
                    "image_id": imid,
                    "category_id": dataset.label_to_coco_label(class_index),
                    "score": float(score),
                    "bbox": bbox.tolist(),
                }
                results.append(image_result)

    if not len(results):
        return

    # write output
    with open(os.path.join(logdir, "val_bbox_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # json.dump(results, open("val_bbox_results.json".format(dataset.set_name), "w"), indent=4)

    # load results in COCO evaluation tool
    coco_true = dataset.coco
    coco_pred = coco_true.loadRes(os.path.join(logdir, "val_bbox_results.json"))

    # run COCO evaluation
    coco_eval = COCOeval(coco_true, coco_pred, "bbox")
    coco_eval.params.imgIds = val_image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats

    model.train()

    return stats
