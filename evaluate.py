"""
@File       : evaluate.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2021/1/7
@Desc       : evaluate a trained model on some dataset
"""
import os
import torch
from nets.cddnet import CDDNet3
from utils.utils import load_model
from test_fun.val_cdd import val_cdd
from utils.json_utils import *

"""------------------config model-----------------"""
backbone='vgg16_bn'
# backbone='mobilenet_v2'
model = CDDNet3(num_classes=1,backbone=backbone)
down_ratio=2

# load pretrain model
model_name='CDDNet3_vgg16_SHA_focal_mse0001_r05_test'
model_path=os.path.join('trained_models',model_name,'best.t7')
model = load_model(model, model_path)
model = model.to(0)
model.eval()

# load model info
try:
    model_info_path=os.path.join('trained_models',model_name,'info.json')
    model_info = read_json_model_info(model_info_path)
    MAE=model_info['best_MAE']
    epoch=model_info['best_epoch']
    print("[MODEL INFO]: MAE={:.3f}, epoch={}".format(MAE,epoch))
except:
    epoch=1000
    print("[MODEL INFO] not found")

"""------------------config data------------------"""
data_root_path='data/SHA/test'
image_folder_path=os.path.join(data_root_path,'images')

"""---------------------val--------------------"""
val_cdd(epoch,model,image_folder_path,cfg=None,summary_writer=None,
        down_ratio=down_ratio,short_len=640,score_threshold=0.2)


