"""
@File       : model.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2020/9/21
@Desc       : model process related codes
"""

import sys
sys.path.append('..')
import torch
import torchvision.models as models
from utils.visualize import *
from utils.post_process import ctdet_decode,ctdet_decode_with_scale_map

# get predict bbox
def predict_bbox(model, img_tensor, cfg, threshold, down_ratio=[],use_scale_map=False):
    output = model(img_tensor)[-1]
    pred_hmap = output[0]

    # from heatmap to bbox list: [K,6],x1,y1,x2,y2,score,class
    if not use_scale_map:
        dets = ctdet_decode(*output, K=3000)
    else:
        dets=ctdet_decode_with_scale_map(*output,cfg,K=3000)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

    # translate the coordinate to the input image size
    dets[:, 0] = dets[:, 0] * down_ratio[0]
    dets[:, 1] = dets[:, 1] * down_ratio[1]
    dets[:, 2] = dets[:, 2] * down_ratio[0]
    dets[:, 3] = dets[:, 3] * down_ratio[1]
    dets = dets[:, :5]

    # filter with score sthreshold
    new_box = []
    for i in range(dets.shape[0]):
        if dets[i][4] > threshold:
            new_box.append(dets[i][:5])
    if len(new_box)>0:
        new_box = np.array(new_box)
    else:
        new_box=np.array([[0,0,0,0,0]])

    if use_scale_map:
        return new_box, pred_hmap, output[2]
    else:
        return new_box, pred_hmap


def predict_bbox_and_density(model, img_tensor, cfg, threshold, down_ratio=[],device='gpu'):
    if cfg:
        img_tensor=img_tensor.to(cfg.device)
    else:
        if device=='gpu':
            dev=torch.device('cuda:0')
        else:
            dev=torch.device('cpu')
        img_tensor=img_tensor.to(dev)

    hmap,reg,smap,density = model(img_tensor)[-1]

    # from heatmap to bbox list: [K,6],x1,y1,x2,y2,score,class
    dets = ctdet_decode_with_scale_map(hmap,reg,smap, cfg, K=3000,device=device)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

    # translate the coordinate to the input image size
    dets[:, 0] = dets[:, 0] * down_ratio[0]
    dets[:, 1] = dets[:, 1] * down_ratio[1]
    dets[:, 2] = dets[:, 2] * down_ratio[0]
    dets[:, 3] = dets[:, 3] * down_ratio[1]
    dets = dets[:, :5]

    # filter with score sthreshold
    new_box = []
    for i in range(dets.shape[0]):
        if dets[i][4] > threshold:
            new_box.append(dets[i][:5])
    if len(new_box) > 0:
        new_box = np.array(new_box)
    else:
        new_box = np.array([[0, 0, 0, 0, 0]])

    return new_box, hmap, smap,density



