#!/usr/bin/env python
import torch.nn as nn
import torchvision
import torch
print('import models')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(1200, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class Combined(nn.Module):
    def __init__(self):
        super(Combined, self).__init__()

        self.ocnn = torchvision.models.vgg16(pretrained=True)
        num_ftrs = self.ocnn.classifier[6].out_features
        self.ponn = nn.Sequential(
		nn.Linear(num_ftrs, 512),
		nn.Dropout(),
		nn.Linear(512, 512),
		nn.Linear(512, 24))
        self.pcnn_conv = nn.Sequential(
		nn.Conv2d(3, 96, kernel_size=11, stride=4),
		nn.BatchNorm2d(96),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=2, stride=2),
		nn.Conv2d(96, 256, kernel_size=5, stride=2),
		nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.pcnn_linear = nn.Sequential(
		nn.Linear(9216, 512),
		nn.Dropout(),
		nn.Linear(512, 512),
		nn.Linear(512, 24))
        self.linear = nn.Linear(48, 2)

    def forward(self, x):
        #x=x.view(x.size(0),-1)
        x1 = self.ocnn(x)
        x1 = self.ponn(x1)
        x2 = self.pcnn_conv(x)
        x2 = x2.view(x2.size(0),-1)
        x2 = self.pcnn_linear(x2)
        f_in = torch.cat((x1,x2),1)
        out = self.linear(f_in)
        return out


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model=Combined().to(device)

#for param in model.ocnn.parameters():
#    param.requires_grad = False
