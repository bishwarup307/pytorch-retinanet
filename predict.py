"""
__author__: bishwarup
created: Monday, 28th September 2020 11:06:26 pm
"""

import os
import json
import numpy as np
from tqdm import tqdm
import argparse
import torch

from retinanet.dataloader import ImageDirectory, custom_collate
from retinanet.utils import get_logger
from retinanet import models

logger = get_logger(__name__)
valid_backbones = ('resnet-18', 'resnet-34', 'resnet-50', 'resnet-101', 'resnet-152')

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
    parser.add_argument('--backbone', type = 'str', default = 'resnet-50', help = 'backbone model arch')
    parser.add_argument('--num-class', type = int, help = 'number of classes for the model')
    opt = parser.parse_args()

    if opt.backbone not in valid_backbones:
        raise AttributeError(f'unknown backbone. we only support {', '.join(valid_backbones)}')

    if not opt.output.endswith('.json'):
        raise AttributeError(f'output must be a path to `json` file, got {opt.output}')

    dataset = ImageDirectory(opt.image_dir)
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        collate_fn=custom_collate,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
    if opt.model == 'resnet-18':
        model = model.resnet18(num_class = opt.num_class, pretrained = False)
    elif opt.model == 'resnet-34':
        model = model.resnet34(num_class = opt.num_class, pretrained = False)
    elif opt.model == 'resnet-50':
        model = model.resnet50(num_class = opt.num_class, pretrained = False)
    elif opt.model == 'resnet-101':
        model = model.resnet101(num_class = opt.num_class, pretrained = False)
    elif opt.model == 'resnet-152':
        model = model.resnet152(num_class = opt.num_class, pretrained = False)
    else:
        raise NotImplementedError
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    ckpt = torch.load(opt.weights, map_location=device)
    model.load_state_dict(ckpt.state_dict())
    model.to(device)
    model.eval()
    
    preds = dict()
    for i, (batch, filenames) in tqdm(enumerate(loader), total = len(loader)):
        with torch.no_grad():
            img_id, confs, classes, bboxes = model(batch[0].float().cuda())
        img_id = img_id.cpu().numpy().tolist()
        confs = confs.cpu().numpy()
        classes = classes.cpu().numpy()
        bboxes = bboxes.cpu().numpy().astype(np.int32)
        
        for i, imgid in enumerate(img_id):
            f = filenames[imgid]
            pr = {
                'bbox': bboxes[i].tolist(),
                'confidence': float(confs[i]),
                'class_index': int(classes[i])
            }
            if f in preds:
                preds[f].append(pr)
            else:
                preds[f] = [pr]
    
    with open(opt.output, 'w') as f:
        json.dump(preds,f, indent = 2)