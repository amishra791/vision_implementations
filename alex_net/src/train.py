import argparse

from data import ImageNetDataset
from model import AlexNet

import torch.optim as optim
import torch.nn as nn
import torch

from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, RandomCrop
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import os

SNAPSHOT_DIR_PATH = '/home/amishra/workspace/vision_implementations/alex_net/snapshots' 
TENSORBOARD_PATH = '/home/amishra/workspace/vision_implementations/alex_net/runs'


parser = argparse.ArgumentParser(description='ImageNet classification using AlexNet')
parser.add_argument('--exp_name', type=str, help='name of experiment')
parser.add_argument('--num_workers', type=int, help='number of workers to fetch batch data for training')
parser.add_argument('--gpu', type=str, help='gpu device to use')
parser.add_argument('--shuffle', type=bool, help='whether to shuffle the data after every epoch')



def main():

    args = parser.parse_args()
    device = torch.device('cuda:'+ args.gpu)
    writer = SummaryWriter(os.path.join(TENSORBOARD_PATH, args.exp_name))

    train_dataset = ImageNetDataset('train', transform=transforms.Compose([
                                                                Resize(256),
                                                                RandomCrop(224),
                                                                ToTensor()]))


    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=args.shuffle, collate_fn=ImageNetDataset.collate_fn, num_workers=args.num_workers)
    net = AlexNet()
    net.to(device)
    
    cur_lr = 0.01
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=0.9)


    for epoch in range(23):
        running_loss = 0.0

        for i, data in enumerate(train_dataloader, 0):
            print(i)        
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = net.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and i > 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
                writer.add_scalar(os.path.join(TENSORBOARD_PATH, args.exp_name, 'training_loss'), running_loss/2000, epoch * len(train_dataloader) + i)
                running_loss = 0.0
    
        top_1_error = top_k_error(outputs, labels, 1)
        top_5_error = top_k_error(outputs, labels, 5)
        writer.add_scalar(os.path.join(TENSORBOARD_PATH, args.exp_name, 'top-1'), top_1_error, epoch + 1)
        writer.add_scalar(os.path.join(TENSORBOARD_PATH, args.exp_name, 'top-5'), top_5_error, epoch + 1)
        
        model_name = str(epoch + 1) + '.tar'
        torch.save(net.state_dict(), os.path.join(SNAPSHOT_DIR_PATH, args.exp_name, model_name))
        
        cur_lr = adjust_learning_rate(optimizer, cur_lr, epoch)

    print("Done")


def adjust_learning_rate(optimizer, cur_lr, epoch):
    new_lr = cur_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    return new_lr


def top_k_error(outputs, labels, k):
    with torch.no_grad():
        batch_size = outputs.size()[0]

        _, top_k_classes = torch.topk(outputs, k)
        
        num_correct = 0
        for i in range(batch_size):
            num_correct += labels[i] in top_k_classes[i]

        return 1 - num_correct/float(batch_size)



if __name__ == '__main__':
    main()




