"""
__author__: bishwarup
created: Wednesday, 30th September 2020 6:33:40 pm
"""

import argparse
import os
import json
from tqdm import tqdm
import numpy as np

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from retinanet.dataloader import ImageDirectory, custom_collate
from retinanet.utils import get_logger
from retinanet import model as arch

logger = get_logger(__name__)

parser = argparse.ArgumentParser(description="predict with retinanet model")
parser.add_argument(
    "-i", "--image-dir", type=str, help="path to directory containing inference images"
)
parser.add_argument("-w", "--weights", type=str, help="path to saved checkpoint")
parser.add_argument("-o", "--output-dir", type=str, help="path to output directory")
parser.add_argument("-b", "--batch-size", type=int, default=1, help="batch size for inference")
parser.add_argument("--num-workers", type=int, default=0, help="number of multiprocessing workers")
parser.add_argument("--backbone", type=str, default="resnet-50", help="backbone model arch")
parser.add_argument("--num-class", type=int, help="number of classes for the model")
parser.add_argument(
    "--confidence", type=float, default=0.6, help="minimum confidence for the predictions"
)
parser.add_argument("--nms-threshold", type=float, default=0.5, help="NMS IoU threshold")

#########################
#### dist parameters ###
#########################
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""",
)
parser.add_argument(
    "--world_size",
    default=-1,
    type=int,
    help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""",
)
parser.add_argument(
    "--rank",
    default=0,
    type=int,
    help="""rank of this process:
                    it is set automatically and should not be passed as argument""",
)
parser.add_argument(
    "--local_rank", default=0, type=int, help="this argument is not used and should be ignored"
)


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])

    # prepare distributed
    dist.init_process_group(
        backend="nccl", init_method=args.dist_url, world_size=args.world_size, rank=args.rank,
    )

    # set cuda device
    args.gpu_to_work_on = args.rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu_to_work_on)
    return


def main():
    global opt
    opt = parser.parse_args()
    init_distributed_mode(opt)
    dataset = ImageDirectory(opt.image_dir)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate,
    )

    logger.info("Building data done with {} images loaded.".format(len(dataset)))

    if opt.backbone == "resnet-18":
        model = arch.resnet18(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-34":
        model = arch.resnet34(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-50":
        model = arch.resnet50(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-101":
        model = arch.resnet101(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    elif opt.backbone == "resnet-152":
        model = arch.resnet152(
            num_classes=opt.num_class,
            pretrained=False,
            conf_threshold=opt.confidence,
            nms_iou_threshold=opt.nms_threshold,
        )
    else:
        raise NotImplementedError

    ckpt = torch.load(opt.weights)
    model.load_state_dict(ckpt.state_dict())
    model.cuda()
    model.eval()
    # if opt.rank == 0:
    #     logger.info(model)
    logger.info(f"successfully loaded saved checkpoint.")

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[opt.gpu_to_work_on], find_unused_parameters=True,
    )

    for i, (batch, filenames) in tqdm(enumerate(loader), total=len(loader)):
        preds = dict()
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

        for img_filename, detection in preds.items():
            with open(os.path.join(opt.output_dir, img_filename.replace("jpg", "json")), "w") as f:
                json.dump(detection, f, indent=2)


if __name__ == "__main__":
    main()
