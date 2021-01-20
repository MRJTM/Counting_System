"""
@File       : MobileNetV2.py
@Author     : Zhijie Cap
@Email      : dearzhijie@gmail.com
@Date       : 2020/12/28
@Desc       : mobilenet v2 and mobilenet v2 backbone
"""

import torch.nn as nn
import torchvision.models as models
from torch.utils import model_zoo
import math
import sys
import os
sys.path.append(os.getcwd())
from nets.blocks import BaseConv

__all__ = ['MobileNetV2']

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, expansion=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes*expansion)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=inplanes*expansion)
        self.bn2 = nn.BatchNorm2d(inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MobileNetV2(nn.Module):

    def __init__(self, block_num_list, num_classes=1000):
        self.inplanes = 32
        super(MobileNetV2, self).__init__()
        self.conv1 = BaseConv(3,32,kernel=3,stride=2,use_bn=True)
        self.bottleneck1 = self._make_bottleneck(Bottleneck, 16, block_num_list[0], stride=1, expansion=1)
        self.bottleneck2 = self._make_bottleneck(Bottleneck, 24, block_num_list[1], stride=2, expansion=6)
        self.bottleneck3 = self._make_bottleneck(Bottleneck, 32, block_num_list[2], stride=2, expansion=6)
        self.bottleneck4 = self._make_bottleneck(Bottleneck, 64, block_num_list[3], stride=2, expansion=6)
        self.bottleneck5 = self._make_bottleneck(Bottleneck, 96, block_num_list[4], stride=1, expansion=6)
        self.bottleneck6 = self._make_bottleneck(Bottleneck, 160, block_num_list[5], stride=2, expansion=6)
        self.bottleneck7 = self._make_bottleneck(Bottleneck, 320, block_num_list[6], stride=1, expansion=6)

        self.conv8 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv9 = nn.Conv2d(1280,num_classes, kernel_size=1, stride=1, bias=False)

        self.random_initialize()

    def random_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_bottleneck(self, block, planes, blocks, stride, expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

            bottlenecks = []
            bottlenecks.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
            self.inplanes = planes
            for i in range(1, blocks):
                bottlenecks.append(block(self.inplanes, planes, expansion=expansion))

            return nn.Sequential(*bottlenecks)

    def load_pretrain_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        pass

    def forward(self, x):
        x = self.conv1(x)

        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)

        x = self.conv8(x)
        x = self.avgpool(x)
        x = self.conv9(x)
        x = x.view(x.size(0),-1)

        return x

class MobileNetV2_backbone(nn.Module):

    def __init__(self, block_num_list=[1,2,3,4,3,3,1], output_layer_list=[1,2,4,7,11]):
        self.inplanes = 32
        super(MobileNetV2_backbone, self).__init__()
        self.output_layer_list=output_layer_list
        self.bottleneck_num=len(block_num_list)

        self.conv1 = BaseConv(3,32,kernel=3,stride=2,use_bn=True)
        if self.bottleneck_num >= 1:
            self.bottleneck1 = self._make_bottleneck(Bottleneck, 16, block_num_list[0], stride=1, expansion = 1)
        if self.bottleneck_num >= 2:
            self.bottleneck2 = self._make_bottleneck(Bottleneck, 24, block_num_list[1], stride=2, expansion = 6)
        if self.bottleneck_num >= 3:
            self.bottleneck3 = self._make_bottleneck(Bottleneck, 32, block_num_list[2], stride=2, expansion = 6)
        if self.bottleneck_num >= 4:
            self.bottleneck4 = self._make_bottleneck(Bottleneck, 64, block_num_list[3], stride=2, expansion = 6)
        if self.bottleneck_num >= 5:
            self.bottleneck5 = self._make_bottleneck(Bottleneck, 96, block_num_list[4], stride=1, expansion = 6)
        if self.bottleneck_num >= 6:
            self.bottleneck6 = self._make_bottleneck(Bottleneck, 160, block_num_list[5], stride=2, expansion = 6)
        if self.bottleneck_num >= 7:
            self.bottleneck7 = self._make_bottleneck(Bottleneck, 320, block_num_list[6], stride=1, expansion = 6)

        self.random_initialize()

    def forward(self, x):
        output_feature_list = []
        x = self.conv1(x)
        if 1 in self.output_layer_list:
            output_feature_list.append(x)

        if self.bottleneck_num >= 1:
            x = self.bottleneck1(x)
            if 2 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 2:
            x = self.bottleneck2(x)
            if 4 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 3:
            x = self.bottleneck3(x)
            if 7 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 4:
            x = self.bottleneck4(x)
            if 11 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 5:
            x = self.bottleneck5(x)
            if 14 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 6:
            x = self.bottleneck6(x)
            if 17 in self.output_layer_list:
                output_feature_list.append(x)

        if self.bottleneck_num >= 7:
            x = self.bottleneck7(x)
            if 18 in self.output_layer_list:
                output_feature_list.append(x)

        return output_feature_list

    def random_initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_pretrain_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_bottleneck(self, block, planes, blocks, stride, expansion):

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

        bottlenecks = []
        bottlenecks.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            bottlenecks.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*bottlenecks)



