"""
@File       : backbones.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/12/25
@Desc       :
"""

import torchvision.models as models
from nets.VGG16 import VGG16_backbone
from nets.MobileNetV2 import MobileNetV2_backbone

CDDNet2_backbone_dict={
    'vgg16_bn':models.vgg16_bn(pretrained=True),
    'mobilenet_v2':models.mobilenet_v2(pretrained=True)
}

CDDNet3_backbone_dict={
    'vgg16_bn':VGG16_backbone(),
    'mobilenet_v2':MobileNetV2_backbone()
}

backbone_feature_layer_dict={
    'vgg16_bn':[5,12,22,32,42],
    'mobilenet_v2':[0,1,3,6,10]
}

class getLayerOutput:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()