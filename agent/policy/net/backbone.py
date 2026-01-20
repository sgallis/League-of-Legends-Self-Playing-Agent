import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    def __init__(self, in_channels):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = ResBlock(in_channels, in_channels, stride=1)
        self.block2 = ResBlock(in_channels, in_channels, stride=1)
        self.block3 = ResBlock(in_channels, 4*in_channels, stride=2)
        self.block4 = ResBlock(4*in_channels, 4*in_channels, stride=1)
        self.block5 = ResBlock(4*in_channels, 4*4*in_channels, stride=2)
        self.block6 = ResBlock(4*4*in_channels, 4*4*in_channels, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        # 3x256x256 -> 32x128x128
        x = F.relu(self.conv1(x), inplace=True)
        # 32x128x128 -> 32x64x64
        x = self.maxpool(x)

        # 32x64x64 -> 32x64x64
        x = self.block1(x)
        # 32x64x64 -> 32x64x64
        x = self.block2(x)
        # 32x64x64 -> 128x32x32
        x = self.block3(x)
        # 128x32x32 -> 128x32x32
        x = self.block4(x)
        # 128x32x32 -> 512x16x16
        x = self.block5(x)
        # 512x16x16 -> 512x16x16
        x = F.relu(self.block6(x))

        # avg pooling
        x = self.avgpool(x).squeeze(dim=(2, 3))
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.stride = stride
        self.shortcut = None
        if self.stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
    
    def forward(self, x):
        x1 = F.relu(x)
        if self.shortcut:
            x = F.relu(x, inplace=True)
            x = self.shortcut(x)
        x1 = F.relu(self.conv1(x1))
        x1 = self.conv2(x1)
        return x + x1
