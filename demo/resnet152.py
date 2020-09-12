import torch
import torch.nn as nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Arguments:
            in_channels: type - int
            out_channels: type - int
            stride: type - int; default - 1
        """
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.batchnorm2d = nn.BatchNorm2d(self.out_channels)
        self.conv2d = nn.Conv2d(self.out_channels, self.out_channels, 3, 11)
 
    def forward(self, x):
        """
        Parameters:
            x: type - FloatTensor; shape - <batch_size, ..., in_channels, in_height, in_width>
        Retvals:
            x: type - FloatTensor; shape - <batch_size, ..., in_channels, in_height, in_width>
        """
        residual = self.batchnorm2d(self.conv2d(F.relu(self.batchnorm2d(self.conv2d(x)))))
        shortcut = self.batchnorm2d(self.conv2d(x))
        return F.relu(residual + shortcut)

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Arguments:
            in_channels: type - int
            out_channels: type - int
            stride: type - int; default - 1
        """
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.batchnorm2d = nn.BatchNorm2d(self.out_channels * 4)
        self.conv2d = nn.Conv2d(self.out_channels, self.out_channels * 4, 1)
 
    def forward(self, x):
        """
        Parameters:
            x: type - FloatTensor; shape - <batch_size, ... in_channels, in_height, in_width>
        Retvals:
            x: type - FloatTensor; shape - <batch_size, ... in_channels, in_height, in_width>
        """
        residual = self.batchnorm2d(self.conv2d(F.relu(self.batchnorm2d(self.conv2d(F.relu(self.batchnorm2d(self.conv2d(x))))))))
        shortcut = self.batchnorm2d(self.conv2d(x))
        return F.relu(residual + shortcut)

