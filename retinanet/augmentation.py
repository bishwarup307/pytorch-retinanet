"""
List of augmenters available:
    * :class:`RandomHorizontalFlip`
    * :class:`RandomRotate`
    * :class:`RandomShear`
    * :class:`RandomBrightnessAdjust`
    * :class:`RandomContrastAdjust`
    * :class:`RandomGammaCorrection`
    * :class:`RandomSaturationAdjust`
    * :class:`RandomHueAdjust`
    * :class:`RandomSharpen`
    * :class:`RandomGaussianBlur`
"""

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

#Referred https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/convolutional.py
class RandomShapren(object):
    """Randomly sharpens the image"""

    def __init__(self, alpha: Optional[Union[Tuple[float,float],float]]=0.5,
                       lightness: Optional[Union[Tuple[float,float],float]]=1.0,
                       p: Optional[float]=0.5):
        """Initialize RandomShapren with alpha and lightness.

        Args:
            alpha (Optional[Union[Tuple[float,float],float]], optional): Blending factor of the sharpened image. At ``0.0``, only the original
        image is visible, at ``1.0`` only its sharpened version is visible. Defaults to 0.5.
            lightness (Optional[Union[Tuple[float,float],float]], optional): Lightness/brightness of the sharpened image. Defaults to 1.0.
            p (Optional[float], optional): Probability of sharpening the image. Defaults to 0.5.
        """        

        if isinstance(alpha,tuple):
            assert len(alpha) == 2, f"if alpha is a tuple, its length should be 2, but got {alpha}."
            assert alpha[0] >= 0. and alpha[0] <= 1. and alpha[1] >= 0. and alpha[1] <= 1.,f"alpha should be in range [0.,1.] but got {alpha}."
            self.alpha = random.uniform(*alpha)
        else:
            self.alpha = alpha

        if isinstance(lightness,tuple):
            assert len(lightness) == 2, f"if lightness is a tuple, its length should be 2, but got {lightness}."
            assert lightness[0] >= 0 and lightness[1] >= 0. ,f"lightness should be in range [0.,None] but got {lightness}."
            self.lightness = random.uniform(*lightness)
        else:
            self.lightness = lightness

        matrix_nochange = np.array([
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ], dtype=np.float32)
        matrix_effect = np.array([
                    [-1, -1, -1],
                    [-1, 8+self.lightness, -1],
                    [-1, -1, -1]
                ], dtype=np.float32)
        self.matrix = ((1-self.alpha) * matrix_nochange + self.alpha * matrix_effect)
        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """Randomly sharpends the image with probablity self.p .

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
        img = cv2.filter2D(img.copy(), -1, self.matrix,dst=img)
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample

#Referred https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/blur.py
class RandomGaussianBlur(object):
    """Randomly Blurrs the image based on guassian kernal standard deviation and size"""

    def __init__(self, sigma: Optional[Union[Tuple[float,float],float]]=0.2,
                       ksize: Optional[int]=None,
                       p: Optional[float]=0.5):
        """Initialize RandomGaussianBlur with sigma(standard deviation) and kernal size.

        Args:
            sigma (Optional[Union[Tuple[float,float],float]], optional): Standard deviation of the gaussian kernel. Defaults to 0.5.
            ksize (Optional[int], optional): Size of the gaussian kernal. Defaults to None.
            p (Optional[float], optional): Probability of blurring the image. Defaults to 0.5
        """              

        if isinstance(sigma,tuple):
            assert len(sigma) == 2, f"if sigma is a tuple, its length should be 2, but got {sigma}."
            assert sigma[0] >= 0. and sigma[0] <= 3. and sigma[1] >= 0. and sigma[1] <= 3.,f"sigma should be in range [0.,3.] but got {sigma}."
            self.sigma = random.uniform(*sigma)
        else:
            self.sigma = sigma

        if ksize is None:
            self.ksize = self._compute_gaussian_blur_ksize(self.sigma)
        else:
            if ksize % 2 == 0:
                ksize += 1
            self.ksize = max(ksize,3)

        self.p = p

    def __call__(self, sample: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
        """Randomly blurs the image using Gaussian kernal with probablity self.p .

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
        img = cv2.GaussianBlur(img.copy(),(self.ksize,self.ksize),sigmaX=self.sigma,sigmaY=self.sigma,dst=img,borderType=cv2.BORDER_REFLECT_101)
        sample = {"img": img.copy(), "annot": bboxes.copy()}
        return sample

    def _compute_gaussian_blur_ksize(self,sigma: float) -> int:
        """Calculates kernal size based on sigma using the formulation mentioned below.
           https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        Args:
            sigma (float): Standard deviation of guassian kernal.

        Returns:
            int: approximated kernal size.
        """        

        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        
        ksize = int(max(ksize, 3))
        if ksize % 2 == 0:
            ksize += 1
        return ksize
    
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
        (RandomShapren,0.,1.),
        (RandomGaussianBlur,0.,2.),
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
               "sharpen":RandomShapren((0.,1.),(0.,2.),p=p),
               "gblur":RandomGaussianBlur((0.,3.),5,p=p),
               }
    return aug_map
