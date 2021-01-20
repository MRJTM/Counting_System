"""
@File       : parse.py
@Author     : caozhijie
@Email      : dearzhijie@gmail.com
@Date       : 2020/6/28
@Desc       : contains functions that parse detection annotations
"""
import sys
import csv
import os
import cv2
from tqdm import tqdm
import scipy
import numpy as np
import scipy.io as io
sys.path.append(os.getcwd())


# 根据point_gt和scale_map,产生bbox_gt
def recover_bbox_from_scale_map(point_list,scale_map):
    bboxs=[]
    h,w=scale_map.shape[0],scale_map.shape[1]
    for p in point_list:
        x,y=int(p[0]),int(p[1])
        y=min(y,scale_map.shape[0]-1)
        s=max(2,scale_map[y,0])
        x1=max(0,x-s/2)
        x2=min(w,x+s/2)
        y1=max(0,y-s/2)
        y2=min(h,y+s/2)
        bboxs.append([x1,y1,x2,y2])
    return bboxs

# 利用smap的方式制作bbox_gt
def parse_SHA_smap(image_folder_path):
    res_dict = {}
    img_names = os.listdir(image_folder_path)
    for i in tqdm(range(len(img_names))):
        img_name=img_names[i]
        img_path = os.path.join(image_folder_path, img_name)

        # load point_gt
        mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('images', 'point_gt').replace('IMG_', 'GT_IMG_'))
        point_list = mat["image_info"][0, 0][0, 0][0]

        # load smap
        scale_map_path = img_path.replace('images', 'scale_map').replace('jpg', 'png')
        scale_map = cv2.imread(scale_map_path)

        # 根据point_gt和scale_map产生bbox
        scale_map = scale_map[:, :, 0]
        box_list= recover_bbox_from_scale_map(point_list, scale_map)

        res_dict[img_name] = []
        for box in box_list:
            res_dict[img_name].append({'bbox': box, 'label': 1})

    return res_dict

parse_func_dict={
    'SHA_smap':parse_SHA_smap,
}