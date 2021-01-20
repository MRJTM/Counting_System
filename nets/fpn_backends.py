"""
@File       : fpn_backends.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/12/25
@Desc       :
"""

import torch
from torch import nn
from torch.utils import model_zoo
from nets.blocks import BaseConv
from nets.VGG16 import VGG16_backbone

class vgg16_BackEnd(nn.Module):
    def __init__(self):
        super(vgg16_BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv5 = BaseConv(256, 64, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv6 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)



    def forward(self, *input):
        conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 = input
        # print("conv1_2.shape:",conv1_2.shape)
        # print("conv2_2.shape:", conv2_2.shape)
        # print("conv3_3.shape:", conv3_3.shape)
        # print("conv4_3.shape:", conv4_3.shape)
        # print("conv5_3.shape:", conv5_3.shape)

        x = self.upsample(conv5_3)          # 1/16->1/8

        x = torch.cat([x, conv4_3], 1)  # 1/8:512+512=1024
        x = self.conv1(x)               # 1024->256
        x4 = self.conv2(x)               # 256->256

        x = self.upsample(x4)            # 1/8->1/4
        x = torch.cat([x, conv3_3], 1)  # 1/4:256+256=512
        x = self.conv3(x)               # 512->128
        x3 = self.conv4(x)               # 128->128

        x = self.upsample(x3)            # 1/4->1/2
        x = torch.cat([x, conv2_2], 1)  # 1/2: 128+128=256
        x = self.conv5(x)               # 256->64
        x2 = self.conv6(x)               # 64->64

        return x2

class mobilenetv2_BackEnd(nn.Module):
    def __init__(self):
        super(mobilenetv2_BackEnd, self).__init__()
        self.conv0 = BaseConv(64, 32, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1 = BaseConv(64, 64, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv2 = BaseConv(64, 24, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv3 = BaseConv(48, 48, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv4 = BaseConv(48, 16, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)

        self.conv5 = BaseConv(32, 32, 1, 1, activation=nn.ReLU(inplace=True), use_bn=True)
        self.conv6 = BaseConv(32, 32, 3, 1, activation=nn.ReLU(inplace=True), use_bn=True)



    def forward(self, *input):
        conv1_2, conv2_2, conv3_3, conv4_3, conv5_3 = input
        # print("conv1_2.shape:",conv1_2.shape)
        # print("conv2_2.shape:", conv2_2.shape)
        # print("conv3_3.shape:", conv3_3.shape)
        # print("conv4_3.shape:", conv4_3.shape)
        # print("conv5_3.shape:", conv5_3.shape)

        x = self.conv0(conv5_3)        # 64->32
        x = self.upsample(x)           # 1/16->1/8

        x = torch.cat([x, conv4_3], 1)  # 1/8:32+32=64
        x = self.conv1(x)               # 64->64
        x4 = self.conv2(x)               # 64->24

        x = self.upsample(x4)            # 1/8->1/4
        x = torch.cat([x, conv3_3], 1)  # 1/4:24+24=48
        x = self.conv3(x)               # 48->48
        x3 = self.conv4(x)               # 48->16

        x = self.upsample(x3)            # 1/4->1/2
        x = torch.cat([x, conv2_2], 1)  # 1/2: 16+16=32
        x = self.conv5(x)               # 32->32
        x2 = self.conv6(x)               # 32->32

        return x2

backend_dict={
    'vgg16_bn':{'model':vgg16_BackEnd(),'out_dim':64},
    'mobilenet_v2':{'model':mobilenetv2_BackEnd(),'out_dim':32}
}