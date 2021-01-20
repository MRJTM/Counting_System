"""
@File       : blocks.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmai.com
@Date       : 2019/10/5
@Desc       : base blocks
"""

import torch.nn as nn
import sys
sys.path.append('../..')

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        pad=dilation*(kernel-1)//2
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=pad,dilation=dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input

class BaseDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, dilation=1, activation=None, use_bn=False):
        super(BaseDeConv, self).__init__()
        pad = kernel // 2
        if dilation > 1:
            pad = dilation
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding=pad,dilation=dilation)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input











