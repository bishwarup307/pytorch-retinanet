from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import logging
import colorama
from typing import Union, Optional, List
import os
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

LOG_COLORS = {
    logging.ERROR: colorama.Fore.RED,
    logging.WARNING: colorama.Fore.YELLOW,
    logging.INFO: colorama.Fore.GREEN,
    logging.DEBUG: colorama.Fore.WHITE,
}


class ColorFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        # if the corresponding logger has children, they may receive modified
        # record, so we want to keep it intact
        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            # we want levelname to be in different color, so let's modify it
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname,
                filename=new_record.filename,
                color_begin=LOG_COLORS[new_record.levelno],
                color_end=colorama.Style.RESET_ALL,
            )
        # now we can let standart formatting take care of the rest
        return super(ColorFormatter, self).format(new_record, *args, **kwargs)


def get_logger(
    name, filepath: Optional[Union[str, os.PathLike]] = None, level: Optional[str] = "debug",
):
    log_level = {"info": logging.INFO, "debug": logging.DEBUG, "error": logging.ERROR}

    logger = logging.getLogger(name)
    logger.setLevel(log_level.get(level.lower(), logging.INFO))
    if filepath is not None:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    formatter = ColorFormatter(
        "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] -> %(message)s"
    )
    if filepath is not None:
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = nn.Parameter(
                torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)), requires_grad=False,
            )
            # if torch.cuda.is_available():
            #     self.mean = torch.nn.Parameters(torch.from_numpy(
            #         np.array([0, 0, 0, 0]).astype(np.float32)
            #     ).cuda()
            # else:
            #     self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            self.std = nn.Parameter(
                torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)),
                requires_grad=False,
            )
            # if torch.cuda.is_available():
            #     self.std = torch.from_numpy(
            #         np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            #     ).cuda()
            # else:
            #     self.std = torch.from_numpy(
            #         np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)
            #     )
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2
        )

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_module(state_dict: OrderedDict):
    clean_state_dict = OrderedDict()
    for key, value in state_dict.items():
        modified_key = key[7:] if key.startswith("module.") else key
        clean_state_dict[modified_key] = value
    return clean_state_dict


class EarlyStopping:
    def __init__(self, wait: int = 10, mode: str = "maximize"):
        if mode == "maximize":
            self.best_metric = -np.inf
        elif mode == "minimize":
            self.best_metric = np.inf
        else:
            raise ValueError("invalid mode specified")
        self.mode = mode
        self.wait = wait
        self._reset_counter()

    def _reset_counter(self):
        self._counter = 0

    def update(self, val):
        if self.wait < 0:
            return False
        if self.mode == "maximize":
            if val >= self.best_metric:
                self._reset_counter()
                self.best_metric = val
            else:
                self._counter += 1
        else:
            if val <= self.best_metric:
                self._reset_counter()
                self.best_metric = val
            else:
                self._counter += 1

        if self._counter >= self.wait:
            return True
        return False


def get_next_run(runs: List[str]):
    if not len(runs):
        return "run0"
    runs = [run for run in runs if run.startswith("run")]
    curr_exp = max([int(x.replace("run", "")) for x in runs])
    return "run" + str(curr_exp + 1)


class CustomSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError("hparam_dict and metric_dict should be dictionary.")
        exp, ssi, sei = hparams(hparam_dict, metric_dict)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)


def get_runtime(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return hour, minutes, seconds


def get_hparams(config):
    hparams = dict()

    # environment params
    hparams["torch_version"] = torch.__version__
    hparams["cuda_availble"] = torch.cuda.is_available()
    hparams["cuda_version"] = torch.version.cuda
    hparams["cudnn_version"] = torch.backends.cudnn.version()
    hparams["n_gpu"] = torch.cuda.device_count()
    hparams["gpu_model"] = torch.cuda.get_device_name(0)

    # dataset params
    hparams["dataset"] = config.dataset
    hparams["image_dir"] = config.image_dir
    hparams["val_image_dir"] = config.val_image_dir
    hparams["train_json_path"] = config.train_json_path
    hparams["val_json_path"] = config.val_json_path
    hparams["image_size"] = config.image_size
    hparams["nsr"] = config.negative_sampling_rate
    hparams["normalize_mean"] = ",".join(list(map(str, config.normalize["mean"])))
    hparams["normalize_std"] = ",".join(list(map(str, config.normalize["std"])))
    hparams["logdir"] = config.logdir

    # augs params
    hparams["hflip"] = config.augs["hflip"]
    hparams["vflip"] = config.augs["vflip"]
    hparams["color_jitter"] = config.augs["color_jitter"]
    hparams["brightness"] = config.augs["brightness"]
    hparams["contrast"] = config.augs["contrast"]
    hparams["gamma"] = config.augs["gamma"]
    hparams["sharpness"] = config.augs["sharpness"]
    hparams["gaussian_blur"] = config.augs["gaussian_blur"]
    hparams["superpixels"] = config.augs["superpixels"]
    hparams["additive_noise"] = config.augs["additive_noise"]
    hparams["shiftscalerotate"] = config.augs["shiftscalerotate"]
    hparams["perspective"] = config.augs["perspective"]
    hparams["rgb_shift"] = (
        ",".join(list(map(str, config.augs["rgb_shift"]))) if config.augs["rgb_shift"] else None
    )
    hparams["cutout"] = (
        ",".join(list(map(str, config.augs["cutout"]))) if config.augs["cutout"] else None
    )
    hparams["min_visibility"] = config.augs["min_visibility"]
    hparams["min_area"] = config.augs["min_area"]

    # model params
    hparams["depth"] = config.backbone
    hparams["pretrained"] = config.pretrained
    hparams["from_checkpoint"] = config.weights
    hparams["freeze_bn"] = config.freeze_bn
    hparams["focal_alpha"] = config.alpha
    hparams["focal_gamma"] = config.gamma

    # learning params
    hparams["num_epochs"] = config.num_epochs
    hparams["batch_size"] = config.batch_size
    hparams["workers"] = config.workers
    hparams["optimizer"] = config.optimizer
    hparams["lr_schedule"] = config.lr_schedule
    hparams["base_lr"] = config.base_lr
    hparams["final_lr"] = config.final_lr
    hparams["warmup_epochs"] = config.warmup_epochs
    hparams["start_warmup"] = config.start_warmup
    hparams["weight_decay"] = config.weight_decay
    hparams["early_stopping"] = config.early_stopping

    return hparams
