import argparse
import collections
import json
import math
import os
from os import confstr
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from config import Config
from retinanet import coco_eval
from retinanet import model
from retinanet.dataloader import (
    CocoDataset,
    collater,
    eval_collate,
    AspectRatioBasedSampler,
    stack_labels,
)
from retinanet.larc import LARC
from retinanet.utils import (
    get_logger,
    EarlyStopping,
    get_hparams,
    get_next_run,
    get_runtime,
    CustomSummaryWriter,
    AverageMeter,
)

assert torch.__version__.split(".")[0] == "1"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def init_distributed_mode(args):
    """
    Initialize the following variables:
        - world_size
        - rank
    """
    if not dist.is_available():
        return

    args.is_slurm_job = "SLURM_JOB_ID" in os.environ

    if args.is_slurm_job:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NNODES"]) * int(
            os.environ["SLURM_TASKS_PER_NODE"][0]
        )
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
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


def load_checkpoint(model: nn.Module, weights: str) -> nn.Module:
    """Loads already trained weights to initialized model.

    Args:
        model (nn.Module): Empty retinanet model
        weights (str): Path to checkpoint
        depth (int): ResNet depth

    Raises:
        KeyError: If current model and checkpoint layers are not matching.

    Returns:
        nn.Module : retinanet model.
    """
    if weights is None:
        return model
    if weights.endswith(".pt"):  # pytorch format
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            ckpt = {
                k: v for k, v in ckpt.state_dict().items() if model.state_dict()[k].shape == v.shape
            }
            model.load_state_dict(ckpt, strict=True)
            logger.info("Resuming training from checkpoint in {}".format(weights))
        except KeyError as e:
            raise e
        del ckpt
        return model
    return model


def parse():
    parser = argparse.ArgumentParser(
        description="Simple training script for training a RetinaNet network."
    )
    # parser.add_argument("--dataset", help="Dataset type, must be one of csv or coco.")
    # parser.add_argument("--train-json-path", help="Path to COCO directory")
    # parser.add_argument("--val-json-path", help="Path to COCO directory")
    # parser.add_argument("--image-dir", help="Path to the images")
    # parser.add_argument(
    #     "--val-image-dir", type=str, help="path to validation images", required=False
    # )
    # parser.add_argument(
    #     "--csv_train", help="Path to file containing training annotations (see readme)"
    # )
    # parser.add_argument(
    #     "--csv_classes", help="Path to file containing class list (see readme)"
    # )
    # parser.add_argument(
    #     "--csv_val",
    #     help="Path to file containing validation annotations (optional, see readme)",
    # )
    #
    # parser.add_argument(
    #     "--depth",
    #     help="Resnet depth, must be one of 18, 34, 50, 101, 152",
    #     type=int,
    #     default=50,
    # )
    # parser.add_argument(
    #     "--resize",
    #     type=str,
    #     help="training dimension of images, specify width and height separated by a comma if they differ, defaults to 512,512",
    #     default="512",
    # )
    # parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
    # parser.add_argument("--batch-size", type=int, help="batch_size", default=8)
    # parser.add_argument(
    #     "--num-workers", type=int, help="number of workers for dataloader mp", default=0
    # )
    # parser.add_argument(
    #     "--logdir", type=str, help="path to save the logs and checkpoints"
    # )
    #
    # parser.add_argument(
    #     "--plot", action="store_true", help="whether to plot images in tensorboard"
    # )
    # parser.add_argument(
    #     "--nsr",
    #     type=float,
    #     default=None,
    #     help="whether to use negative sampling of images",
    # )
    #
    # parser.add_argument(
    #     "--augs",
    #     help="available augs:rand,hflip,rotate,shear,brightness,contrast,hue,gamma,saturation,sharpen,gblur should be seperated by spaces.",
    #     nargs="+",
    # )
    # parser.add_argument(
    #     "--augs-prob",
    #     type=float,
    #     help="probability of applying augmentation in range [0.,1.]",
    # )

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
        "--local_rank", default=0, type=int, help="this argument is not used and should be ignored",
    )
    # parser.add_argument(
    #     "--base_lr", default=0.001, type=float, help="base learning rate"
    # )
    # parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    # parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    # parser.add_argument(
    #     "--warmup_epochs", default=10, type=int, help="number of warmup epochs"
    # )
    # parser.add_argument(
    #     "--start_warmup", default=0, type=float, help="initial warmup learning rate"
    # )

    parser.add_argument(
        "--dist-mode",
        type=str,
        choices=["DP", "DDP"],
        default="DDP",
        help="whether to use DataParallel or DistributedDataParallel",
    )
    # parser.add_argument(
    #     "--weights", default="", type=str, help="model weights path to resume training"
    # )

    return parser


def parse_resize(image_size: int) -> List[int]:
    # dims = resize_str.split(",")
    if not hasattr(image_size, "len"):
        image_size = [image_size, image_size]
    return image_size


def validate(model, dataset, valid_loader):
    model.eval()
    cls_loss, reg_loss = [], []

    for i, (images, labels, scales, offset_x, offset_y, image_ids) in tqdm(
        enumerate(valid_loader), total=len(valid_loader), leave=keep_pbar
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

    model.train()
    return np.mean(cls_loss), np.mean(reg_loss)


def main():
    global args, results, val_image_ids, logger, hparams

    args = parse().parse_args()
    hparams = get_hparams(Config)

    try:
        logdir = Config.logdir
        prev_runs = os.listdir(logdir) if os.path.isdir(logdir) else []
        curr_run = get_next_run(prev_runs)
        logdir = os.path.join(logdir, curr_run)
        Path(logdir).mkdir(parents=True, exist_ok=True)
        print(f"logdir configured as {logdir}")
        # os.makedirs(logdir, exist_ok=True)
    except Exception as exc:
        raise exc

    log_file = os.path.join(logdir, "train.log")
    logger = get_logger(__name__, log_file)

    try:
        init_distributed_mode(args)
        distributed = True
    except KeyError:
        args.rank = 0
        distributed = False

    if args.dist_mode == "DP":
        distributed = True
        args.rank = 0

    if args.rank == 0:
        start = time.perf_counter()
        logger.info(f"distributed mode: {args.dist_mode if distributed else 'OFF'}")
        logger.info(f"pretrained weights for backbone: {Config.pretrained}")

    if Config.val_image_dir is None:
        if args.rank == 0:
            logger.info(
                "No validation image directory specified, will assume the same image directory for train and val"
            )
        Config.val_image_dir = Config.image_dir

    writer = CustomSummaryWriter(log_dir=logdir)
    img_dim = parse_resize(Config.image_size)

    if args.rank == 0:
        logger.info(f"training image dimensions: {img_dim[0]},{img_dim[1]}")
        logger.info("CUDA available: {}".format(torch.cuda.is_available()))
        logger.info(f"torch.__version__ = {torch.__version__}")

    # Create the data loaders
    if Config.dataset == "coco":

        augs = Config.augs
        normalize = Config.normalize
        dataset_train = CocoDataset(
            image_dir=Config.image_dir,
            json_path=Config.train_json_path,
            image_size=img_dim,
            normalize=normalize,
            transform=augs,
            return_ids=False,
        )

    else:
        raise ValueError("Dataset type not understood (must be csv or coco), exiting.")

    if dist.is_available() and distributed and args.dist_mode == "DDP":
        sampler = DistributedSampler(dataset_train)
        dataloader_train = DataLoader(
            dataset_train,
            sampler=sampler,
            batch_size=Config.batch_size,
            num_workers=Config.workers,
            collate_fn=collater,
            shuffle=True,
        )

    elif Config.negative_sampling_rate is not None:
        logger.info(
            f"using WeightedRandomSampler with negative (image) sample rate = {Config.negative_sampling_rate}"
        )
        weighted_sampler = WeightedRandomSampler(
            dataset_train.weights, len(dataset_train), replacement=True
        )
        dataloader_train = DataLoader(
            dataset_train,
            num_workers=Config.workers,
            collate_fn=collater,
            sampler=weighted_sampler,
            batch_size=Config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    else:
        sampler = AspectRatioBasedSampler(
            dataset_train, batch_size=Config.batch_size, drop_last=False
        )
        dataloader_train = DataLoader(
            dataset_train,
            num_workers=Config.workers,
            collate_fn=collater,
            # shuffle=True,
            batch_sampler=sampler,
            pin_memory=True,
        )

    if Config.val_json_path is not None:
        dataset_val = CocoDataset(
            Config.val_image_dir,
            Config.val_json_path,
            image_size=img_dim,
            normalize=Config.normalize,
            return_ids=True,
        )
    else:
        dataset_val = None

    depth = int(Config.backbone.split("-")[-1])
    # Create the model
    if depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes,
            pretrained=Config.pretrained,
            nms_iou_threshold=Config.nms_iou_threshold,
            conf_threshold=Config.conf_threshold,
            alpha=Config.alpha,
            gamma=Config.gamma,
        )
    elif depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes,
            pretrained=Config.pretrained,
            nms_iou_threshold=Config.nms_iou_threshold,
            conf_threshold=Config.conf_threshold,
            alpha=Config.alpha,
            gamma=Config.gamma,
        )
    elif depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes,
            pretrained=Config.pretrained,
            nms_iou_threshold=Config.nms_iou_threshold,
            conf_threshold=Config.conf_threshold,
            alpha=Config.alpha,
            gamma=Config.gamma,
        )
    elif depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes,
            pretrained=Config.pretrained,
            nms_iou_threshold=Config.nms_iou_threshold,
            conf_threshold=Config.conf_threshold,
            alpha=Config.alpha,
            gamma=Config.gamma,
        )
    elif depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes,
            pretrained=Config.pretrained,
            nms_iou_threshold=Config.nms_iou_threshold,
            conf_threshold=Config.conf_threshold,
            alpha=Config.alpha,
            gamma=Config.gamma,
        )
    else:
        raise ValueError(
            "Unsupported backbone specified, deppth must be one of 18, 34, 50, 101, 152"
        )

    # Load checkpoint if provided.
    retinanet = load_checkpoint(retinanet, Config.weights)

    if torch.cuda.is_available():
        if dist.is_available() and distributed:
            if args.dist_mode == "DDP":
                retinanet = nn.SyncBatchNorm.convert_sync_batchnorm(retinanet)
                retinanet = retinanet.cuda()
            elif args.dist_mode == "DP":
                retinanet = torch.nn.DataParallel(retinanet).cuda()
            else:
                raise NotImplementedError
        else:
            torch.cuda.set_device(torch.device("cuda:0"))
            retinanet = retinanet.cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(
    #     retinanet.parameters(), lr=4.2, momentum=0.9, weight_decay=1e-4,
    # )

    if dist.is_available() and distributed and args.dist_mode == "DDP":
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=True)

    warmup_lr_schedule = np.linspace(
        Config.start_warmup, Config.base_lr, len(dataloader_train) * Config.warmup_epochs,
    )
    iters = np.arange(len(dataloader_train) * (Config.num_epochs - Config.warmup_epochs))
    cosine_lr_schedule = np.array(
        [
            Config.final_lr
            + 0.5
            * (Config.base_lr - Config.final_lr)
            * (
                1
                + math.cos(
                    math.pi
                    * t
                    / (len(dataloader_train) * (Config.num_epochs - Config.warmup_epochs))
                )
            )
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    if distributed and dist.is_available() and args.dist_mode == "DDP":
        retinanet = nn.parallel.DistributedDataParallel(
            retinanet, device_ids=[args.gpu_to_work_on], find_unused_parameters=True
        )

    loss_hist = collections.deque(maxlen=500)

    if dist.is_available() and distributed:
        retinanet.module.train()
        if Config.freeze_bn:
            retinanet.module.freeze_bn()
    else:
        retinanet.train()
        if Config.freeze_bn:
            retinanet.freeze_bn()

    if args.rank == 0:
        logger.info("Number of training images: {}".format(len(dataset_train)))
        if dataset_val is not None:
            logger.info("Number of validation images: {}".format(len(dataset_val)))

    # scaler = amp.GradScaler()
    global best_map
    best_map = 0

    scaler = amp.GradScaler(enabled=True)
    global keep_pbar
    keep_pbar = not (distributed and args.dist_mode == "DDP")

    early_stopping = EarlyStopping(wait=Config.early_stopping, mode="minimize")

    stop = False
    map_avg, map_50, map_75, map_small = 0, 0, 0, 0
    try:
        for epoch in range(Config.num_epochs):
            cls_loss = AverageMeter()
            reg_loss = AverageMeter()
            torch.cuda.empty_cache()
            if stop:
                break
            if dist.is_available() and distributed:
                if args.dist_mode == "DDP":
                    dataloader_train.sampler.set_epoch(epoch)
                retinanet.module.train()
                if Config.freeze_bn:
                    retinanet.module.freeze_bn()
            else:
                retinanet.train()
                if Config.freeze_bn:
                    retinanet.freeze_bn()
            # retinanet.module.freeze_bn()

            epoch_loss = []
            results = []
            val_image_ids = []

            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), leave=keep_pbar,)
            for iter_num, data in pbar:
                cur_batch_size = data["img"].size(0)

                n_iter = epoch * len(dataloader_train) + iter_num

                for param_group in optimizer.param_groups:
                    lr = lr_schedule[n_iter]
                    param_group["lr"] = lr

                optimizer.zero_grad()

                if torch.cuda.is_available():
                    with amp.autocast(enabled=False):
                        classification_loss, regression_loss = retinanet(
                            [data["img"].cuda().float(), data["annot"].cuda()]
                        )
                else:
                    classification_loss, regression_loss = retinanet(
                        [data["img"].float(), data["annot"]]
                    )

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                cls_loss.update(classification_loss.item(), cur_batch_size)
                reg_loss.update(regression_loss.item(), cur_batch_size)

                loss = classification_loss + regression_loss

                if args.rank == 0:
                    writer.add_scalar("Learning rate", lr, n_iter)
                pbar_desc = f"Epoch: {epoch} | steps: {n_iter} |lr = {lr:0.6f} | batch: {iter_num} | cls: {cls_loss.avg:.4f} | reg: {reg_loss.avg:.4f}"
                pbar.set_description(pbar_desc)
                pbar.update(1)
                if bool(loss == 0):
                    continue

                # loss.backward()
                scaler.scale(loss).backward()

                # unscale the gradients for grad clipping
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                # optimizer.step()
                # scheduler.step()  # one cycle lr operates at batch level
                scaler.step(optimizer)
                scaler.update()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                del classification_loss
                del regression_loss

            if Config.dataset == "coco":
                if len(dataset_val) > 0:
                    if dist.is_available() and distributed and args.dist_mode == "DDP":
                        sampler_val = DistributedSampler(dataset_val)
                        dataloader_val = DataLoader(
                            dataset_val,
                            sampler=sampler_val,
                            batch_size=Config.batch_size,
                            num_workers=Config.workers,
                            collate_fn=eval_collate,
                            pin_memory=True,
                        )
                    else:
                        dataloader_val = DataLoader(
                            dataset_val,
                            batch_size=Config.batch_size,
                            num_workers=Config.workers,
                            collate_fn=eval_collate,
                            pin_memory=True,
                            drop_last=False,
                        )
                else:
                    dataloader_val = None

                if dataloader_val is not None:
                    val_cls_loss, val_reg_loss = validate(retinanet, dataset_val, dataloader_val)
                else:
                    val_cls_loss, val_reg_loss = -1, -1

                if args.rank == 0:
                    if len(results):
                        with open(os.path.join(logdir, "val_bbox_results.json"), "w") as f:
                            json.dump(results, f, indent=4)
                        stats = coco_eval.evaluate_coco(dataset_val, val_image_ids, logdir)
                        map_avg, map_50, map_75, map_small = stats[:4]
                    else:
                        map_avg, map_50, map_75, map_small = [-1] * 4

                    if map_50 > best_map:
                        best_map = map_50
                    torch.save(
                        retinanet.state_dict(),
                        os.path.join(
                            logdir,
                            f"retinanet_{Config.backbone.replace('-', '_')}_epoch_{epoch}.pt",
                        ),
                    )

                    stop = early_stopping.update(val_cls_loss)

                    writer.add_scalar("eval/cls_loss", val_cls_loss, epoch * len(dataloader_train))
                    writer.add_scalar("eval/reg_loss", val_reg_loss, epoch * len(dataloader_train))

                    writer.add_scalar(
                        "eval/map@0.5:0.95", map_avg, epoch * len(dataloader_train),
                    )
                    writer.add_scalar("eval/map@0.5", map_50, epoch * len(dataloader_train))
                    writer.add_scalar("eval/map@0.75", map_75, epoch * len(dataloader_train))
                    writer.add_scalar(
                        "eval/map_small", map_small, epoch * len(dataloader_train),
                    )
                    logger.info(
                        f"Epoch: {epoch} | lr = {lr:.6f} |map@0.5:0.95 = {map_avg:.4f} | map@0.5 = {map_50:.4f} | map@0.75 = {map_75:.4f} | map-small = {map_small:.4f}"
                    )
                    logger.info(
                        f"cls_loss_val: {val_cls_loss:.4f}, reg_loss_val: {val_reg_loss:.4f}"
                    )

            else:
                raise ValueError("`dataset` is not COCO")
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print(f"map_avg: {map_avg:.4f}")
        print(f"map_50: {map_50:.4f}")
        print(f"map_75: {map_75:.4f}")
        print(f"map_small: {map_small:.4f}")

        if args.rank == 0:
            runtime = int(time.perf_counter() - start)
            h, m, s = get_runtime(runtime)
            writer.add_hparams(
                hparam_dict=hparams,
                metric_dict={
                    "mAP_avg": map_avg,
                    "mAP_50": best_map,
                    "mAP_75": map_75,
                    "mAP_small": map_small,
                },
            )
            logger.info(f"total runtime: {h}h {m}m {s}s")
            with open(os.path.join(logdir, "hparams.json"), "w") as f:
                json.dump(hparams, f, indent=2)
    retinanet.eval()


if __name__ == "__main__":
    main()
