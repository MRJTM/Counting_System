"""
@File       : VGG16.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2019/11/17
@Desc       : VGG16 backbone
"""

import torch.nn as nn
from torchvision import models
from nets.blocks import BaseConv
from torch.utils import model_zoo

# VGG16 backbone
class VGG16_backbone(nn.Module):
    def __init__(self):
        super(VGG16_backbone, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, input):
        input = self.conv1_1(input)
        conv1_2 = self.conv1_2(input)   # 1:64
        input = self.pool(conv1_2)        # 1->1/2 downsample

        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)   # 1/2:128
        input = self.pool(conv2_2)      # 1/4 downsample

        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)   # 1/4:256
        input = self.pool(conv3_3)      # 1/8 downsample

        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)   # 1/8: 512
        input = self.pool(conv4_3)      # 1/16 downsample

        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)   # 1/16: 512

        return conv1_2, conv2_2, conv3_3, conv4_3, conv5_3

    def load_pretrain_weight(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = \
                state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.load_state_dict(new_dict)