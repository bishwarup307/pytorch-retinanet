from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import cv2
import glob
from typing import List,Tuple,Dict,Optional,Union
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate
from pycocotools.coco import COCO
from retinanet.bbox_utils import *
import torchvision.transforms.functional as TF

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, image_dir, json_path, transform=None, return_ids=False):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.root_dir = root_dir
        # self.set_name = set_name
        self.image_dir = image_dir
        self.transform = transform

        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()
        self.return_ids = return_ids
        self.load_classes()
        print(f"number of classes: {self.num_classes}")

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c["id"]
            self.coco_labels_inverse[c["id"]] = len(self.classes)
            self.classes[c["name"]] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
        # print(len(self.labels))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)

        if self.return_ids:
            return sample, self.image_ids[idx]

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info["file_name"])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a["bbox"][2] < 1 or a["bbox"][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a["bbox"]
            annotation[0, 4] = self.coco_label_to_label(a["category_id"])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image["width"]) / float(image["height"])

    @property
    def num_classes(self):
        return len(self.labels)


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=","))
        except ValueError as e:
            raise (ValueError("invalid CSV class file: {}: {}".format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(
                    csv.reader(file, delimiter=","), self.classes
                )
        except ValueError as e:
            raise (ValueError("invalid CSV annotations file: {}: {}".format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, "rb")
        else:
            return open(path, "r", newline="")

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError("line {}: format should be 'class_name,class_id'".format(line)))
            class_id = self._parse(class_id, int, "line {}: malformed class ID: {{}}".format(line))

            if class_name in result:
                raise ValueError("line {}: duplicate class name: '{}'".format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a["x1"]
            x2 = a["x2"]
            y1 = a["y1"]
            y2 = a["y2"]

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a["class"])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(
                    ValueError(
                        "line {}: format should be 'img_file,x1,y1,x2,y2,class_name' or 'img_file,,,,,'".format(
                            line
                        )
                    ),
                    None,
                )

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ("", "", "", "", ""):
                continue

            x1 = self._parse(x1, int, "line {}: malformed x1: {{}}".format(line))
            y1 = self._parse(y1, int, "line {}: malformed y1: {{}}".format(line))
            x2 = self._parse(x2, int, "line {}: malformed x2: {{}}".format(line))
            y2 = self._parse(y2, int, "line {}: malformed y2: {{}}".format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError(
                    "line {}: x2 ({}) must be higher than x1 ({})".format(line, x2, x1)
                )
            if y2 <= y1:
                raise ValueError(
                    "line {}: y2 ({}) must be higher than y1 ({})".format(line, y2, y1)
                )

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError(
                    "line {}: unknown class name: '{}' (classes: {})".format(
                        line, class_name, classes
                    )
                )

            result[img_file].append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "class": class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s["img"] for s in data]
    annots = [s["annot"] for s in data]
    scales = [s["scale"] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, : int(img.shape[0]), : int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, : annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {"img": padded_imgs, "annot": annot_padded, "scale": scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=512, max_side=512):
        image, annots = sample["img"], sample["annot"]

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows * scale)), int(round((cols * scale))))
        )
        rows, cols, cns = image.shape

        pad_w = (32 - rows % 32) % 32
        pad_h = (32 - cols % 32) % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {
            "img": torch.from_numpy(new_image),
            "annot": torch.from_numpy(annots),
            "scale": scale,
        }

#Taken from https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/data_aug.py    
class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*
    """

    def __init__(self, p: Optional[float] = 0.5):
        """
        Initalize RandomHorizontalFlip with probablity of flipping p.

        Args:
            p (Optional[float]): Probability of flipping. Defaults to 0.5.
        """          
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly flip the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """                          
        img, bboxes = sample["img"], sample["annot"]
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))
        if random.random() < self.p:
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
        
#Taken from https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/data_aug.py 
class RandomRotate(object):
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    """

    def __init__(self, angle: Optional[Union[Tuple[int,int],int]]=20,p: Optional[float]=0.5,random_select: Optional[bool]=True):
        """
        Initalize RandomRotate with angle and probablity of flipping p.
        Args:
            angle (Optional[Union[Tuple[int,int],int]], optional): Angle or Range of angles by which image to be rotated. Defaults to 20.
            p (Optional[float], optional): Probability of rotation. Defaults to 0.5.
            random_select (Optional[bool], optional): Whether to randomly select the angle or use provided one. Defaults to True.
        """        
        self.angle = angle
        self.p = p
        self.random_select = random_select
        
        if self.random_select:
            if type(self.angle) == tuple:
                assert len(self.angle) == 2, "Invalid range"  
            
            else:
                self.angle = (-self.angle, self.angle)
            
    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly rotate the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """                        
    
        img, bboxes = sample["img"], sample["annot"]
        
        if random.random() > self.p:
            return sample
            
        if self.random_select:
            angle = random.uniform(*self.angle)
        else:
            angle = self.angle
    
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
    
        img = rotate_im(img, angle)
    
        corners = get_corners(bboxes)
    
        corners = np.hstack((corners, bboxes[:,4:]))
    
    
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
    
        new_bbox = get_enclosing_box(corners)
    
    
        scale_factor_x = img.shape[1] / w
    
        scale_factor_y = img.shape[0] / h
    
        img = cv2.resize(img, (w,h))
    
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    
        bboxes  = new_bbox
    
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
    
#Taken from https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/data_aug.py 
class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    """

    def __init__(self, shear_factor: Optional[Union[Tuple[float,float],float]] = 0.2,
                        p: Optional[float]=0.5,
                        random_select: Optional[bool]=True):
        """
        Initialize RandomShear with shear factor and probablity of shearing with p.
        Args:
            shear_factor (Optional[Union[Tuple[float,float],float]], optional): shear factor or Range of shear factors by which image to be sheared. Defaults to 0.2.
            p (Optional[float], optional): Probability of shearing. Defaults to 0.5.. Defaults to 0.5.
            random_select (Optional[bool], optional):  Whether to randomly select the shear factor or use provided one. Defaults to True.
        """        
        
        self.shear_factor = shear_factor
        self.p = p
        self.random_select = random_select
        
        if self.random_select:
            if type(self.shear_factor) == tuple:
                assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
            else:
                self.shear_factor = (-self.shear_factor, self.shear_factor)
        
    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly shear the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """    
        
        if random.random() > self.p:
            return sample
    
        img,bboxes = sample["img"], sample["annot"]
        
        if self.random_select:
            shear_factor = random.uniform(*self.shear_factor)
        else:
            shear_factor = self.shear_factor
    
        w,h = img.shape[1], img.shape[0]
        
        if shear_factor < 0:
            sample = RandomHorizontalFlip(p=1.)(sample)
            
        img,bboxes = sample["img"], sample["annot"]
        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
    
        nW =  img.shape[1] + abs(shear_factor*img.shape[0])
    
        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 
    
    
        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))
    
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        if shear_factor < 0:
            sample = RandomHorizontalFlip(p=1.)(sample)
            
        img,bboxes = sample["img"], sample["annot"]
        img = cv2.resize(img, (w,h))
    
        scale_factor_x = nW / w
    
        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 
    
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
    
class RandomBrightnessAdjust(object):
    """Adjust brightness of an Image."""

    def __init__(self, brightness_factor: Optional[float]=0.5,p: Optional[float]=0.5):
        """
        Initialize RandomBrightnessAdjust with brighness factor and  probablity of adjustment with p.
        Args:
            brightness_factor (Optional[float], optional): brighness factor to be applied on image. Defaults to 0.5.
            p (Optional[float], optional): Probability of adjusting brighness. Defaults to 0.5.
        """        
        self.brightness_factor = brightness_factor
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly adjust brighness of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """    
        
        if random.random() > self.p:
            return sample
        
        img,bboxes = sample["img"], sample["annot"]
        img = img*255.
        #print(f"In Brighness: Factor={self.brightness_factor}")
        img = img.astype(np.uint8)
        img = TF.to_tensor(img)
        img = TF.to_pil_image(img,mode="RGB")
        img = TF.adjust_brightness(img, self.brightness_factor)
        img = TF.pil_to_tensor(img)
        img = img.permute(1,2,0).numpy().astype(np.float32)
        img = img/255.
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
    
class RandomContrastAdjust(object):
    """Adjust contrast of an Image."""

    def __init__(self, contrast_factor: Optional[float]=0.5,p: Optional[float]=0.5):
        """
        Initialize RandomContrastAdjust with contrast factor and  probablity of adjustment with p.
        Args:
            contrast_factor (Optional[float], optional): contrast factor to be applied on image. Defaults to 0.5.
            p (Optional[float], optional): Probability of adjusting contrast. Defaults to 0.5.
        """ 
        self.contrast_factor = contrast_factor
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly adjust contrast of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """     
        
        if random.random() > self.p:
            return sample
        
        img,bboxes = sample["img"], sample["annot"]
        img = img*255.
        #print(f"In Contrast: Factor={self.contrast_factor}")
        img = img.astype(np.uint8)
        img = TF.to_tensor(img)
        img = TF.to_pil_image(img,mode="RGB")
        img = TF.adjust_contrast(img, self.contrast_factor)
        img = TF.pil_to_tensor(img)
        img = img.permute(1,2,0).numpy().astype(np.float32)
        img = img/255.
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
class RandomGammaCorrection(object):
    """Perform gamma correction on an image."""

    def __init__(self, gamma: Optional[float]=0.2,p: Optional[float]=0.5):
        """
        Initialize RandomGammaCorrection with gamma factor and  probablity of correction with p.
        Args:
            gamma (Optional[float], optional): gamma correction factor to be applied on image. Defaults to 0.2. Range is [-0.5,0.5]
            p (Optional[float], optional): Probability of correcting gamma. Defaults to 0.5.
        """ 
        self.gamma = gamma
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly correct gamma factor of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """    
        
        if random.random() > self.p:
            return sample
        
        img,bboxes = sample["img"], sample["annot"]
        img = img*255.
        #print(f"In Gamma: Factor={self.gamma}")
        img = img.astype(np.uint8)
        img = TF.to_tensor(img)
        img = TF.to_pil_image(img,mode="RGB")
        img = TF.adjust_gamma(img, self.gamma)
        img = TF.pil_to_tensor(img)
        img = img.permute(1,2,0).numpy().astype(np.float32)
        img = img/255.
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
    
class RandomSaturationAdjust(object):
    """Adjust color saturation of an image."""

    def __init__(self, saturation_factor: Optional[float]=0.5,p: Optional[float]=0.5):
        """
        Initialize RandomSaturationAdjust with saturation factor and  probablity of correction with p.
        Args:
            saturation_factor (Optional[float], optional): saturation factor to be applied on image. Defaults to 0.2.
            p (Optional[float], optional): Probability of adjusting saturation. Defaults to 0.5.
        """
        self.saturation_factor = saturation_factor
        self.p = p 

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly adjust saturation factor of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """    
        
        if random.random() > self.p:
            return sample
        
        img,bboxes = sample["img"], sample["annot"]
        img = img*255.
        #print(f"In Saturation: P={self.p},Factor={self.saturation_factor}")
        img = img.astype(np.uint8)
        img = TF.to_tensor(img)
        img = TF.to_pil_image(img,mode="RGB")
        img = TF.adjust_saturation(img, self.saturation_factor)
        img = TF.pil_to_tensor(img)
        img = img.permute(1,2,0).numpy().astype(np.float32)
        img = img/255.
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
        
class RandomHueAdjust(object):
    """Adjust hue of an image."""

    def __init__(self, hue_factor: Optional[float]=0.5,p: Optional[float]=0.5):
        """
        Initialize RandomHueAdjust with hue factor and  probablity of correction with p.
        Args:
            hue_factor (Optional[float], optional): hue factor to be applied on image. Defaults to 0.2.
            p (Optional[float], optional): Probability of adjusting hue. Defaults to 0.5.
            
        """
        self.hue_factor = hue_factor
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly adjust hue factor of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """   
        
        if random.random() > self.p:
            return sample
        
        img,bboxes = sample["img"], sample["annot"]
        img = img*255.
        #print(f"In Hue: Factor={self.hue_factor}")
        img = img.astype(np.uint8)
        img = TF.to_tensor(img)
        img = TF.to_pil_image(img,mode="RGB")
        img = TF.adjust_hue(img, self.hue_factor)
        img = TF.pil_to_tensor(img)
        img = img.permute(1,2,0).numpy().astype(np.float32)
        img = img/255.
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample
    
def augment_list() -> List:
    """
    Get all available augmentation in a list.
    Returns:
        List: list of all available augmentations.
    """    
    l = [(RandomBrightnessAdjust,0.1,1.1),
        (RandomHorizontalFlip, 0.5, 0.5),
        (RandomRotate, -30, 30),
        (RandomShear, -0.2, 0.2),        
        (RandomContrastAdjust,0.1,1.1),
        (RandomGammaCorrection,0.05,0.95),
        (RandomSaturationAdjust,0.1,1.1),
        (RandomHueAdjust,-0.5,0.5),
        ]
    
    return l
    
        
 #Taken from https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py   
class RandAugment(object):
    def __init__(self, n: int, m: int, p: Optional[float] = 0.7):
        """
        Initialize base on number of augmentations to be applied and m to determine augmentation value.
        Args:
            n (int): Number of augmentations to be applied.
            m (int): To determine augmentation factor to be applied. Need to be in range [0, 30]
            p (Optional[float], optional): Probability of applying augmentation. Defaults to 0.7.
        """        
        self.n = n
        self.m = m
        self.p = p
        self.augment_list = augment_list()

    def __call__(self,sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """
        Randomly adjust hue factor of the image with probablity self.p . 

        Args:
            sample (Dict[str,np.ndarray]): image and bounding boxes to be augmented in the format 
                                           {"img":np.ndarray,"annot":np.ndarray}

        Returns:
            Dict[str,np.ndarray]: augmented image and bounding boxes in the format 
                                  {"img":np.ndarray,"annot":np.ndarray}
        """

        ops = random.choices(self.augment_list, k=self.n)
        #print(ops)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            if op in [RandomRotate,RandomShear]:
                aug = op(val,p=self.p,random_select=False)
            elif op in [RandomHorizontalFlip]:
                aug = op(val)
            else:
                aug = op(val,p=self.p)
            sample = aug(sample)
        return sample

class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample["img"], sample["annot"]
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {"img": image, "annot": annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample["img"], sample["annot"]

        return {"img": ((image.astype(np.float32) - self.mean) / self.std), "annot": annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [
            [order[x % len(order)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(order), self.batch_size)
        ]


class ImageDirectory(Dataset):
    def __init__(self, image_dir, ext="jpg"):
        self.images = glob.glob(os.path.join(image_dir, f"*.{ext}"))
        self.transforms = torchvision.transforms.Compose(
            [
                #                   torchvision.transforms.Resize((512, 512)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        return self.transforms(img), os.path.basename(self.images[idx])

    def get_image(self, idx):
        return np.array(Image.open(self.images[idx]))


def custom_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids


def eval_collate(batch):
    image_ids, images, labels, scales = [], [], [], []
    for b in batch:
        instance, img_id = b
        images.append(instance["img"])
        labels.append(instance["annot"])
        scales.append(instance["scale"])
        image_ids.append(img_id)
    return torch.stack(images).permute(0, 3, 1, 2), labels, scales, image_ids


def get_aug_map(p: Optional[int]=0.5) -> Dict[str,object]:
    """
    Get Avaialble augmentation map
    Args:
        p (Optional[int], optional): probability with which augmentation needs to be applied. Defaults to 0.5.

    Returns:
        Dict(str,object): Dictionary of available augmentations.
    """    
    aug_map = {"rand":RandAugment(n=3,m=20,p=max(0.7,p)),
               "hflip":RandomHorizontalFlip(p=p),
               "rotate":RandomRotate(p=p),
               "shear":RandomShear(p=p),
               "brightness":RandomBrightnessAdjust(p=p),
               "contrast":RandomContrastAdjust(p=p),
               "hue":RandomHueAdjust(p=p),
               "gamma":RandomGammaCorrection(p=p),
               "saturation":RandomSaturationAdjust(p=p),
               }
    return aug_map
