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
from torchvision.transforms import ToTensor, Resize, RandomCrop
from skimage import io

CLASS_MAP_PATH = '/home/amishra/datasets/image_net/ILSVRC/devkit/data/map_clsloc.txt'
TRAIN_PATH = '/home/amishra/datasets/image_net/ILSVRC/Data/CLS-LOC/train'
VALIDATION_PATH = '/home/amishra/datasets/image_net/ILSVRC/Data/CLS-LOC/val'


class ImageNetDataset(Dataset):


	def __init__(self, mode, transform=None):
		self.mode = mode

		self.metadata = self.load_imagenet_meta()
		self.data_img_paths, self.data_labels = self.read_image_paths_labels()

                assert len(self.data_img_paths) == len(self.data_labels), 'Number of images is not equal to the number of labels'

                self.transform = transform


	def load_imagenet_meta(self):
		"""
		It reads ImageNet metadata from ILSVRC 2012 dev tool file
		Returns:
			wnids: list of ImageNet wnids labels (as strings)
                        class labels: class labels as numbers (1-1000 since there are 1k classes)
			words: list of words (as strings) referring to wnids labels and describing the classes
		"""
                with open(CLASS_MAP_PATH) as f:

                    parsed_lines = [line.strip().split() for line in f.readlines()]
                    metadata = [tuple(line) for line in parsed_lines]

		return metadata 

	def read_image_paths_labels(self):
		"""
		Reads the paths of the images (from the folders structure)
		and the indexes of the labels (using an annotation file)
                Returns: 
                    paths: list of image paths
                    labels: class labels for each image from [1, NUM_CLASSES]
		"""
		paths = []
		labels = []

		if self.mode == 'train':

		    for wnid, i, _ in self.metadata:
			img_names = os.listdir(os.path.join(TRAIN_PATH, wnid))
			for img_name in img_names:
			    paths.append(os.path.join(TRAIN_PATH, wnid, img_name))
			    labels.append(int(i) - 1) # classes should be 0 <= i < num_classes

		    # shuffling the images names and relative labels
		    d = zip(paths, labels)
		    random.shuffle(d)
		    paths, labels = zip(*d)

		else:
                    for wnid, i, _ in self.metadata:
                        img_names = os.listdir(os.path.join(VALIDATION_PATH, wnid))
                        for img_name in img_names:
                            paths.append(os.path.join(VALIDATION_PATH, wnid, img_name))
                            labels.append(int(i) - 1)

                    d = zip(paths, labels)
                    random.shuffle(d)
                    paths, labels = zip(*d)

		self.dataset_size = len(paths)

		return paths, labels

        def __len__(self):
            return len(self.data_img_paths)

        def __getitem__(self, idx):
            img_path, img_label = self.data_img_paths[idx], self.data_labels[idx]
            image = Image.open(img_path)
            #image = Image.fromarray(image.astype('uint8'), 'RGB')

            if self.transform:
                image = self.transform(image)


            img_label = torch.tensor(img_label)
            if image.size()[0] != 3:
                new_image = Image.new("RGB", image.size()[1:])
                new_image_arr = np.array(new_image)
                new_image_arr = np.rollaxis(new_image_arr, 2)
                image = torch.tensor(new_image_arr, dtype=torch.float)

            #print(idx, len(self.data_img_paths), len(self.data_labels), image.size(), img_label.size())
            return {'image': image, 'label': img_label}
            
        
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
    

