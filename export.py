"""
__author__: bishwarup
created: Thursday, 1st October 2020 8:54:17 pm
"""

from typing import Tuple, Optional
import os
import argparse
import numpy as np
import torch.onnx

from retinanet.utils import get_logger
from retinanet import model

logger = get_logger(__name__)


def export(
    checkpoint: str,
    output_path,
    num_classes: Optional[int] = 1,
    model_arch: Optional[str] = "resnet-50",
    input_size: Optional[Tuple[int, int]] = (512, 512),
    batch_size: Optional[int] = 1,
    verbose: Optional[bool] = False,
):

    assert output_path.endswith(".onnx"), "`output_path` must be path to the output `onnx` file"
    if model_arch == "resnet-18":
        net = model.resnet18(num_classes)
    elif model_arch == "resnet-34":
        net = model.resnet34(num_classes)
    elif model_arch == "resnet-50":
        net = model.resnet50(num_classes)
    elif model_arch == "resnet-101":
        net = model.resnet101(num_classes)
    elif model_arch == "resnet-152":
        net = model.resnet152(num_classes)
    else:
        raise NotImplementedError

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"using device: {device}")
    net = net.to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.info(f"successfully loaded saved checkpoint.")

    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    net.eval()
    net.export = True
    dummy_input = dummy_input.to(device)

    logger.info(f"exporting to {output_path}...")
    torch.onnx.export(
        net,
        dummy_input,
        output_path,
        opset_version=11,
        verbose=verbose,
        input_names=["input"],
        output_names=["anchors", "classification", "regression"],
    )
    logger.info("export complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export pytorch to onnx format")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet-50",
        choices=["resnet-18", "resnet-34", "resnet-50", "resnet-101", "resnet-152"],
        help="model architecture to convert",
    )
    parser.add_argument("--checkpoint", type=str, help="path to saved checkpoint")
    parser.add_argument("--output-path", type=str, help="path to output onnx file")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for export")
    parser.add_argument("--n-classes", type=int, default=1, help="number of classes for the model")
    parser.add_argument("--verbose", action="store_true", help="verbosity of export")
    opt = parser.parse_args()

    export(opt.checkpoint, opt.output_path, opt.n_classes, opt.arch, batch_size=opt.batch_size)
