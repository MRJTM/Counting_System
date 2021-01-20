import os
import sys
sys.path.append(os.getcwd())
from nets.backbones import *
from nets.fpn_backends import *

class CDDNet3(nn.Module):
    def __init__(self,num_classes=1,backbone='vgg16_bn',load_pretrain=False):
        super(CDDNet3, self).__init__()
        self.backbone = CDDNet3_backbone_dict[backbone]
        self.fpn = backend_dict[backbone]['model']
        self.fpn_outdim=backend_dict[backbone]['out_dim']
        print("[backbone]:",backbone)

        if load_pretrain:
            self.backbone.load_pretrain_weight()
            print("pretrain_weight loaded success")

        # heatmap head
        self.hmap_head = nn.Sequential(
            BaseConv(self.fpn_outdim, self.fpn_outdim, 3, activation=nn.ReLU(inplace=True),use_bn=False),
            BaseConv(self.fpn_outdim, self.fpn_outdim//2, 3, activation=nn.ReLU(inplace=True), use_bn=False),
            BaseConv(self.fpn_outdim//2, num_classes, 1, 1, activation=None, use_bn=False)
        )

        # reg head
        self.reg_head=nn.Sequential(
            BaseConv(self.fpn_outdim, self.fpn_outdim, 3, activation=nn.ReLU(inplace=True),use_bn=False),
            BaseConv(self.fpn_outdim, self.fpn_outdim//2, 3, activation=nn.ReLU(inplace=True), use_bn=False),
            BaseConv(self.fpn_outdim//2, 2, 1, 1, activation=None, use_bn=False)
        )

        # w_h_ head
        self.w_h_head=nn.Sequential(
            BaseConv(self.fpn_outdim, self.fpn_outdim, 3, activation=nn.ReLU(inplace=True),use_bn=False),
            BaseConv(self.fpn_outdim, self.fpn_outdim//2, 3, activation=nn.ReLU(inplace=True), use_bn=False),
            BaseConv(self.fpn_outdim//2, 1, 1, 1, activation=None, use_bn=False)
        )

        # density map head
        self.density_head=nn.Sequential(
            BaseConv(self.fpn_outdim, self.fpn_outdim, 3, activation=nn.ReLU(inplace=True), use_bn=False),
            BaseConv(self.fpn_outdim, self.fpn_outdim//2, 3, activation=nn.ReLU(inplace=True), use_bn=False),
            BaseConv(self.fpn_outdim//2, 1, 1, 1, activation=None, use_bn=False)
        )


    def forward(self, input,mode="normal"):
        backbone_features=self.backbone(input)
        fpn_feature=self.fpn(*backbone_features)

        outputs=[]
        hmap=self.hmap_head(fpn_feature)
        reg=self.reg_head(fpn_feature)
        w_h_=self.w_h_head(fpn_feature)
        density=self.density_head(fpn_feature)

        if mode=='feature':
            return fpn_feature
        elif mode=='output_with_feature':
            outputs.append([hmap, reg, w_h_,density,fpn_feature])
            return outputs
        else:
            outputs.append([hmap, reg, w_h_,density])
            return outputs