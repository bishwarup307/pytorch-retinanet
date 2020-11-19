import csv
import glob
import os
import random
import sys
from typing import Tuple, Dict, Optional, List, Sequence
import albumentations as A

import cv2
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler


def _transform_image(sample: Dict, transforms: Dict):

    transforms = {k: v for k, v in transforms.items() if v}
    if len(transforms) <= 2:
        return sample

    augs = []
    for name, params in transforms.items():
        # print(name)
        if name == "hflip":
            augs.append(A.HorizontalFlip(p=params))
        if name == "vflip":
            augs.append(A.VerticalFlip(p=params))
        if name == "shiftscalerotate":
            augs.append(A.ShiftScaleRotate(p=params))
        if name == "gamma":
            augs.append(A.RandomGamma(p=params))
        if name == "sharpness":
            augs.append(A.IAASharpen(p=params))
        if name == "gaussian_blur":
            augs.append(A.GaussianBlur(p=params))
        if name == "superpixels":
            augs.append(A.IAASuperpixels(p=params))
        if name == "additive_noise":
            augs.append(A.IAAAdditiveGaussianNoise(p=params))
        if name == "perspective":
            augs.append(A.IAAPerspective(p=params))
        if name == "color_jitter":
            augs.append(A.ColorJitter(p=params))
        if name == "brightness":
            augs.append(A.RandomBrightness(p=params))
        if name == "contrast":
            augs.append(A.RandomContrast(p=params))

        if name == "rgb_shift":
            augs.append(
                A.RGBShift(
                    r_shift_limit=params[0],
                    g_shift_limit=params[1],
                    b_shift_limit=params[2],
                    p=params[3],
                )
            )
        if name == "cutout":
            augs.append(
                A.Cutout(max_h_size=params[0], max_w_size=params[1], p=params[2])
            )
    trsf = A.Compose(
        augs,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=transforms["min_visibility"],
            min_area=transforms["min_area"],
        ),
    )
    # print(sample["img"].dtype)
    transformed = trsf(
        image=sample["img"],
        bboxes=sample["annot"][:, :-1],
        category_ids=sample["annot"][:, -1],
    )
    if len(transformed["bboxes"]):
        annot = np.concatenate(
            [
                np.array(transformed["bboxes"]),
                np.array(transformed["category_ids"])[..., np.newaxis],
            ],
            axis=1,
        )
    # print(annot.shape)
    sample["img"] = transformed["image"]
    sample["annot"] = annot

    return sample


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(
        self,
        image_dir: str,
        json_path: str,
        image_size: Sequence,
        normalize: Optional[Dict] = None,
        transform: Optional[Dict] = None,
        return_ids: bool = False,
        nsr: float = None,
    ):
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
        self.image_size = image_size
        try:
            self.normalize_mean = normalize["mean"]
            self.normalize_std = normalize["std"]
        except TypeError:
            self.normalize_mean, self.normalize_std = None, None

        self.coco = COCO(json_path)
        self.image_ids = self.coco.getImgIds()
        self.return_ids = return_ids
        self.nsr = nsr if nsr is not None else 1.0
        self.load_classes()
        self._obtain_weights()
        print(f"number of classes: {self.num_classes}")

    def _obtain_weights(self):
        weights = []
        for imid in self.image_ids:
            anns = self.coco.getAnnIds([imid])
            if anns:
                weights.append(1)
            else:
                weights.append(self.nsr)
        self.weights = weights

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

    def _to_tensor(self, sample, normalize=True):
        if normalize:
            normalizer = Normalizer(self.normalize_mean, self.normalize_std)
            sample["img"] = normalizer(sample["img"])

        sample["img"] = torch.from_numpy(sample["img"].astype(np.float32))
        sample["annot"] = torch.from_numpy(sample["annot"].astype(np.float32))
        return sample

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx, normalize=False)  # load image
        annot = self.load_annotations(idx)
        sample = {"img": img, "annot": annot}
        if self.image_size is not None:
            resize = Resizer(self.image_size)  # resize
            sample = resize(sample)

        if self.transform is not None:
            sample = _transform_image(sample, self.transform)

        if self.return_ids:
            return self._to_tensor(sample), self.image_ids[idx]

        return self._to_tensor(sample)

    def load_image(self, image_index, normalize=True):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.image_dir, image_info["file_name"])
        img = np.array(Image.open(path))

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        if not normalize:
            return img

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False
        )
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
            raise (
                ValueError("invalid CSV class file: {}: {}".format(self.class_list, e))
            )

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
            raise (
                ValueError(
                    "invalid CSV annotations file: {}: {}".format(self.train_file, e)
                )
            )
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
                raise (
                    ValueError(
                        "line {}: format should be 'class_name,class_id'".format(line)
                    )
                )
            class_id = self._parse(
                class_id, int, "line {}: malformed class ID: {{}}".format(line)
            )

            if class_name in result:
                raise ValueError(
                    "line {}: duplicate class name: '{}'".format(line, class_name)
                )
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

            result[img_file].append(
                {"x1": x1, "x2": x2, "y1": y1, "y2": y2, "class": class_name}
            )
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
    # scales = [s["scale"] for s in data]

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

    padded_imgs = padded_imgs.permute(0, 3, 1, 2).contiguous()

    return {"img": padded_imgs, "annot": annot_padded}


def letterbox(image, expected_size, fill_value=0):
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    # print(image)
    new_img = np.full((eh, ew, 3), fill_value, dtype=np.uint8)
    # fill new image with the resized image and centered it

    offset_x, offset_y = (ew - nw) // 2, (eh - nh) // 2

    new_img[offset_y : offset_y + nh, offset_x : offset_x + nw, :] = image.copy()
    return new_img, scale, offset_x, offset_y


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, annots = sample["img"], sample["annot"]
        # print(f"image dtype before resizer: {image.dtype}")
        rsz_img, scale, offset_x, offset_y = letterbox(image, self.size)

        annots[:, :4] *= scale
        annots[:, 0] += offset_x
        annots[:, 1] += offset_y
        annots[:, 2] += offset_x
        annots[:, 3] += offset_y

        return {
            "img": rsz_img,
            "annot": annots,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

        # rows, cols, cns = image.shape

        # smallest_side = min(rows, cols)

        # # rescale the image so the smallest side is min_side
        # scale = min_side / smallest_side

        # # check if the largest side is now greater than max_side, which can happen
        # # when images have a large aspect ratio
        # largest_side = max(rows, cols)

        # if largest_side * scale > max_side:
        #     scale = max_side / largest_side

        # # resize the image with the computed scale
        # image = skimage.transform.resize(
        #     image, (int(round(rows * scale)), int(round((cols * scale))))
        # )
        # rows, cols, cns = image.shape

        # pad_w = (32 - rows % 32) % 32
        # pad_h = (32 - cols % 32) % 32

        # new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_image[:rows, :cols, :] = image.astype(np.float32)

        # annots[:, :4] *= scale

        # return {
        #     "img": torch.from_numpy(new_image),
        #     "annot": torch.from_numpy(annots),
        #     "scale": scale,
        # }


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
    def __init__(self, mean: Optional[List[float]], std: Optional[List[float]]):
        self.mean = mean
        self.std = std

    def __call__(self, img: np.ndarray):
        normalized_img = img.astype(np.float32) / 255.0
        if self.mean is not None and self.std is not None:
            normalized_img = (normalized_img - self.mean) / self.std
        return normalized_img


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
    image_ids, images, labels, scales, offset_x, offset_y = [], [], [], [], [], []
    for b in batch:
        instance, img_id = b
        images.append(instance["img"])
        labels.append(instance["annot"])
        scales.append(instance["scale"])
        offset_x.append(instance["offset_x"])
        offset_y.append(instance["offset_y"])
        image_ids.append(img_id)
    return (
        torch.stack(images).permute(0, 3, 1, 2).contiguous(),
        labels,
        scales,
        offset_x,
        offset_y,
        image_ids,
    )
