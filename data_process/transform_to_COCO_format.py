"""
@File       : transform_to_COCO_format.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2020/8/16
@Desc       : generate COCO detection ground truth file
"""

"""
COCO detection format：
images: 
    0001.jpg
    0002.jpg
    ...

annotations:
    train.json
    val.json

"""

import os
import sys
import cv2
import json
from tqdm import tqdm
sys.path.append(os.getcwd())
print("sys.path:",sys.path)
from data_process.parse import parse_func_dict

"""--------------------------config--------------------------------"""
train_or_test='test'
data_root_path='D:/dataset/SHA/{}'.format(train_or_test)
image_folder_path=os.path.join(data_root_path,'images')
COCO_gt_folder_path=os.path.join(data_root_path,'annotations')
if not os.path.exists(COCO_gt_folder_path):
    os.mkdir(COCO_gt_folder_path)

# class name list
class_list=['head']
class_to_int={}
for i,class_name in enumerate(class_list):
    class_to_int[class_name]=i+1

# choose parse function
parse_func=parse_func_dict['SHA_smap']

# set json file save path
json_file_path=os.path.join(COCO_gt_folder_path,'head_gt.json')
print("COCO annotation files will be saved into:",json_file_path)

print("\n-------------------------parse raw annotation----------------------------")
# raw_anno_dict:{'img_name1':[{'bbox':[x1,y1,x2,y2],'label':1},{'bbox':[x1,y1,x2,y2],'label':2},,,], 'img_path2':[{},{},,,]}
raw_anno_dict=parse_func(image_folder_path)

print("\n-------------------------generating COCO annotations------------------------")
categories=[]
for i,class_name in enumerate(class_list):
    catogory={'id':i+1,'name':class_name,'superatogory':'restricted_obj'}
    categories.append(catogory)

res_dict={'images':[],'annotations':[],'categories':categories}
img_id=0
anno_id=0
img_names=list(raw_anno_dict.keys())
total_img_num=len(img_names)
for i in tqdm(range(total_img_num)):
    # write image info
    img_name=img_names[i]
    img_path=os.path.join(image_folder_path,img_name)
    img=cv2.imread(img_path)
    H=img.shape[0]
    W=img.shape[1]

    img_dict = {
        'file_name': img_name,
        'id': img_id,
        'height': H,
        'width': W
    }
    res_dict['images'].append(img_dict)

    # write annotation info
    anno_list=raw_anno_dict[img_name]
    for k,anno in enumerate(anno_list):
        x1,y1,x2,y2=anno['bbox']
        width=x2-x1
        height=y2-y1
        area=width*height
        category_id=anno['label']

        anno_dict={
            'id':anno_id,
            'image_id':img_id,
            'bbox':[x1,y1,width,height],
            'area':area,
            'category_id':category_id,
            'segmentation':[],
            'iscrowd':0,
        }

        res_dict['annotations'].append(anno_dict)

        anno_id+=1

    img_id += 1

# save annotation files
with open(json_file_path, 'w') as f:
    json.dump(res_dict, f)