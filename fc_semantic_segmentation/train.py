import argparse

from dataset import ModifiedCityscapes


import torch.optim as optim
import torch.nn as nn
import torch

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, RandomCrop
from torch.utils.data import DataLoader

from torchvision.datasets import Cityscapes

from tensorboardX import SummaryWriter
import os

SNAPSHOT_DIR_PATH = '/home/amishra/workspace/vision_implementations/fc_semantic_segmentation/snapshots' 
TENSORBOARD_PATH = '/home/amishra/workspace/vision_implementations/fc_semantic_segmentation/runs'


parser = argparse.ArgumentParser(description='Semantic segmentation using FC nets')
parser.add_argument('--exp_name', type=str, help='name of experiment')
parser.add_argument('--num_workers', type=int, help='number of workers to fetch batch data for training')
parser.add_argument('--gpu', type=str, help='gpu device to use')
parser.add_argument('--shuffle', type=bool, help='whether to shuffle the data after every epoch')




def main():

    train_dataset = ModifiedCityscapes('/home/amishra/datasets/cityscapes', split='train', mode='fine', target_type='color')


    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=ModifiedCityscapes.collate_fn, num_workers=2)
    
    num_epochs = 1

    for k in range(num_epochs):

        
        for i, data in enumerate(train_dataloader):

            images, labels = data
            print((images.size(), labels.size()))


if __name__ == '__main__':
    main()
