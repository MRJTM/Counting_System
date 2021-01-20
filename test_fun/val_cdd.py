"""
@File       : val_ccd.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2020/9/23
@Desc       : val CDDNet
"""
import os
import cv2
import numpy as np
import torch
from utils.image import resize_img_with_short_len,img_to_tensor,resize_gt
from utils.model import predict_bbox_and_density

def val_cdd(epoch,model,image_folder_path,cfg,summary_writer,down_ratio,short_len,score_threshold):
    print('\n Val@Epoch: %d' % epoch)
    model.eval()
    torch.cuda.empty_cache()
    image_folder_name=image_folder_path.split(os.path.sep)[-1]
    img_names = os.listdir(image_folder_path)
    img_names.sort()
    total_img_num = len(img_names)
    if cfg:
        split_num=int(total_img_num*cfg.val_split_ratio)
    else:
        split_num=total_img_num
    gap=total_img_num//split_num
    new_img_names=[img_names[k] for k in range(0,total_img_num,gap)]
    img_names=new_img_names
    MAE=0
    MSE=0
    den_MAE=0
    det_MAE=0
    scale_threshold=12

    for i, img_name in enumerate(img_names):
        # if i%50==0:
        #     print("val finished {}/{}".format(i,len(img_names)))
        img_path = os.path.join(image_folder_path, img_name)
        # print("----[{}/{}]:{}----".format(i,len(img_names),img_name))
        img = cv2.imread(img_path)

        gt_num_path = img_path.replace(image_folder_name, 'gt_num').replace('jpg', 'npy')
        gt_num = np.load(gt_num_path)

        """resize img to short len"""
        img = resize_img_with_short_len(img, short_len=short_len)
        height, width = img.shape[0], img.shape[1]

        """test"""
        with torch.no_grad():
            """turn img to tensor"""
            img = img_to_tensor(img)
            if cfg:
                img = img.to(cfg.device)
            else:
                img = img.to(0)

            """model forward"""
            ratio = [down_ratio,down_ratio]
            preds = predict_bbox_and_density(model, img, cfg, threshold=score_threshold,down_ratio=ratio)
            pred_bbox=preds[0]
            pred_smap=preds[2]
            pred_smap = pred_smap.detach().cpu().numpy().reshape(pred_smap.shape[2], pred_smap.shape[3])
            pred_density=preds[3]
            pred_density = pred_density.detach().cpu().numpy().reshape(pred_density.shape[2], pred_density.shape[3])
            pred_density = resize_gt(pred_density, height, width)

            """resize scale map to the img size"""
            smap1 = pred_smap * down_ratio
            smap1 = cv2.resize(smap1, (width, height))

            """get the density map part"""
            pred_density_part=pred_density.copy()
            pred_density_part[np.where(smap1 >= scale_threshold)]=0
            density_part_num=np.sum(pred_density_part)
            den_MAE+=np.abs(np.sum(pred_density)-gt_num)

            """get the bbox part"""
            new_box_list = []
            for j in range(pred_bbox.shape[0]):
                x1, y1, x2, y2, score = pred_bbox[j]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                if smap1[center_y, center_x] >= scale_threshold:
                    new_box_list.append([x1, y1, x2, y2])
            bbox_part_num=len(new_box_list)
            det_MAE+=np.abs(pred_bbox.shape[0]-gt_num)

            """MAE，MSE"""
            pred_num=density_part_num+bbox_part_num
            AE=np.abs(pred_num-gt_num)
            SE=np.square(pred_num-gt_num)
            MAE += AE
            MSE += SE

            print("[{}/{}]:{}: den_num={:.3f},det_num={:d},gt_num={:.3f}".format(i,len(img_names),
                                                                       img_name,
                                                                       np.sum(pred_density),
                                                                       pred_bbox.shape[0],
                                                                       gt_num))

    # MAE=MAE/len(img_names)
    MSE=np.sqrt(MSE/len(img_names))
    den_MAE/=len(img_names)
    det_MAE/=len(img_names)
    # MAE=(den_MAE+det_MAE)/2
    MAE=det_MAE

    print("\n[Final]: den_MAE={:.3f},det_MAE={:.3f},combine_MAE={:.3f},MSE={:.3f}".format(den_MAE,det_MAE,MAE,MSE))
    if summary_writer:
        summary_writer.add_scalar('val/MAE', MAE, epoch)
        summary_writer.add_scalar('val/MSE', MSE, epoch)

    return MAE