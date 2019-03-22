"""
Imagenet dataset class
"""
import os
import random
from PIL import Image
import scipy.ndimage, scipy.misc
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import Cityscapes
from torchvision.transforms import ToTensor, Resize, RandomCrop
from skimage import io

class ModifiedCityscapes(Cityscapes):

        def __init__(self, root_dir, split, mode, target_type):
            super().__init__(root=root_dir, split=split, mode=mode, target_type=target_type)


        def __getitem__(self, idx):
            img, color = super().__getitem__(idx)
            img = torch.tensor(np.array(img))
            color = torch.tensor(np.array(color))
            
            if color.size()[2] != 3:
                new_color = Image.new("RGB", color.size()[:-1])
                new_color_arr = np.array(new_color)
                new_color_arr = np.rollaxis(new_color_arr, 1, 0)
                #new_color_arr = np.rollaxis(new_color_arr, 2)
                color = torch.tensor(new_color_arr, dtype=torch.float)

            return {'image': img, 'label': color}
            
        
        @staticmethod
        def collate_fn(lst):
            '''Takes in a list of samples by tupling and outputs a list of samples. 
               This is used for the Dataloader class during training
            '''
            images = [cur_sample['image'] for cur_sample in lst]
            labels = [cur_sample['label'] for cur_sample in lst]

            images = torch.stack(images)
            labels = torch.stack(labels)

            return images, labels
