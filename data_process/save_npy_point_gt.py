"""
@File       : save_npy_point_gt.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2021/1/5
@Desc       : change the raw head point annotation into uniform numpy array
            : The output is a numpy array: [[x1,y1],[x2,y2]...]
"""

import os
import numpy as np
from tqdm import tqdm
import scipy.io as io

"""--------------config--------------"""
data_root_path='D:\dataset\SHA'
part_name='test'
point_gt_folder_path=os.path.join(data_root_path,part_name,'point_gt')
npy_point_gt_folder_path=os.path.join(data_root_path,part_name,'npy_point_gt')
os.makedirs(npy_point_gt_folder_path,exist_ok=True)

"""--------------Process-----------"""
file_names=os.listdir(point_gt_folder_path)
for i in tqdm(range(len(file_names))):
    old_gt_path=os.path.join(point_gt_folder_path,file_names[i])
    new_gt_path=os.path.join(npy_point_gt_folder_path,file_names[i].replace('GT_','').replace('mat','npy'))
    old_gt=io.loadmat(old_gt_path)["image_info"][0, 0][0, 0][0]
    np.save(new_gt_path,old_gt)

