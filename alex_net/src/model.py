import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, stride=4, padding=2)
        self.conv1.bias.data.fill_(0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(96, 256, 5, padding=2, groups=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.conv4 = nn.Conv2d(384, 384, 3, padding=1, groups=2)        
        self.conv5 = nn.Conv2d(384, 256, 3, padding=1, groups=2)
        self.pool5 = nn.MaxPool2d(3, stride=2)


        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 1000)
        

    def forward(self, x):
        x= self.conv1(x)
        x= self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool5(x)

        x = x.view(-1, reduce((lambda x, y: x * y), x.size()[1:]))

        x = self.fc6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.dropout7(x)
        x = self.fc8(x)


        return x




