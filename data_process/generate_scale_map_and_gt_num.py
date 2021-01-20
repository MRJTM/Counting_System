"""
@File       : generate_scale_map_and_gt_num.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2021/1/5
@Desc       : generate scale map from npy_point_gt, and save gt_num at the same time
"""

import os
import cv2
import scipy
import scipy.ndimage
import scipy.spatial
import numpy as np
import sys

sys.path.append(os.getcwd())
from utils.visualize import *


# input 2-D point gt map, output gt density map
def generate_scale(gt_list, max_size=48):
    scale_list = []
    gt_count = len(gt_list)
    if gt_count == 0:
        return scale_list

    pts = np.array(gt_list)
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    # put gaussian kernel to each point
    for i, pt in enumerate(pts):
        if gt_count > 1:
            head_size = (distances[i][1] + distances[i][2] + distances[i][3]) / 3
            head_size = min(head_size, max_size)
        else:
            head_size = max_size
        scale_list.append([pt[0], pt[1], head_size])

    return scale_list

def turn_scale_to_box(scale_list, w, h):
    box_list = []
    for scale in scale_list:
        x, y, s = scale[0], scale[1], scale[2]
        x1 = max(0, x - s / 2)
        x2 = min(w, x + s / 2)
        y1 = max(0, y - s / 2)
        y2 = min(h, y + s / 2)
        box_list.append([x1, y1, x2, y2])
    box_list = np.array(box_list)
    return box_list

def generate_scale_map_by_fit_from_box(bboxs, w, h):
    # generate data pair [y,s]
    y_list = []
    s_list = []
    for i in range(bboxs.shape[0]):
        x1, y1, x2, y2 = bboxs[i]
        y_list.append((y1 + y2) / 2)
        s_list.append(min(x2 - x1, y2 - y1))
    y = np.array(y_list)
    s = np.array(s_list)

    # linear fit
    fit_fun = np.polyfit(y, s, 1)
    h_list = np.arange(h)
    new_s_list = np.polyval(fit_fun, h_list)

    # generate scale map
    scale_map = np.zeros((h, w), np.int)
    for i in range(h):
        scale_map[i, :] = max(0, new_s_list[i])
    return scale_map, h_list, new_s_list


"""------------------config-----------------"""
data_root_path='D:\dataset\SHA'
part_name='train'
image_folder_path=os.path.join(data_root_path,part_name,'images')
point_gt_folder_path=os.path.join(data_root_path,part_name,'npy_point_gt')

"""-----------------process------------------"""
from tqdm import tqdm

scale_map_folder_path = os.path.join(data_root_path, part_name,'scale_map')
gt_num_folder_path = os.path.join(data_root_path,part_name, 'gt_num')
if not os.path.exists(scale_map_folder_path):
    os.mkdir(scale_map_folder_path)
if not os.path.exists(gt_num_folder_path):
    os.mkdir(gt_num_folder_path)

save_gt_num = True

img_names = os.listdir(image_folder_path)
img_names.sort()
for i in tqdm(range(len(img_names))):
    img_name = img_names[i]
    img_path = os.path.join(image_folder_path, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]

    """get point gt"""
    point_gt_name=img_name.replace('jpg','npy')
    point_gt_path=os.path.join(point_gt_folder_path,point_gt_name)
    point_list=np.load(point_gt_path)

    """get bbox from point gt by geometry-adaptive"""
    scale_list = generate_scale(point_list, max_size=50)
    bboxs = turn_scale_to_box(scale_list, w, h)

    gt_num = point_list.shape[0]
    if save_gt_num:
        gt_num_save_path = os.path.join(gt_num_folder_path, img_name.replace('jpg', 'npy'))
        np.save(gt_num_save_path, gt_num)

    """generate scale map by linear fit"""
    scale_map, h_list, new_s_list = generate_scale_map_by_fit_from_box(bboxs, w, h)

    """save scale map"""
    scale_map_save_path = os.path.join(scale_map_folder_path, img_name.replace('jpg', 'png'))
    cv2.imwrite(scale_map_save_path, scale_map)

