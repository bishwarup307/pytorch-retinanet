import sys
import os
import torch
import numpy as np
import random
import cv2
from typing import List,Tuple,Dict,Optional,Union
from retinanet.bbox_utils import *
import torchvision.transforms.functional as TF

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