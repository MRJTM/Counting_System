import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data

from utils.image import color_aug
from utils.image import draw_umich_gaussian
from utils.image import resize_img_with_short_len

SHA_NAMES = ['__background__', 'head']

SHA_IDS = [1]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


class SHA(data.Dataset):
    def __init__(self, data_dir, split, split_ratio=1.0, img_size=512,
                 down_ratio=4,radius_ratio=0.5):
        super(SHA, self).__init__()
        self.num_classes = 1
        self.class_name = SHA_NAMES
        self.valid_ids = SHA_IDS
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}  # [1:0]

        self.data_rng = np.random.RandomState(123)  # 设置伪随机数种子，为了使代码容易复现
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]
        self.radius_ratio=radius_ratio

        self.split = split
        print("data_dir:", data_dir)
        self.data_dir = data_dir

        self.img_dir = os.path.join(self.data_dir, '%s' % split, 'images')
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'test', 'annotations', 'head_gt.json')
        else:
            self.annot_path = os.path.join(self.data_dir, 'train', 'annotations', 'head_gt.json')

        self.max_objs = 10000
        self.padding = 127  # 31 for resnet/resdcn
        self.down_ratio = down_ratio
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.gaussian_iou = 0.7
        self.short_len = 896  # 图像短边resize到什么尺寸

        print('==> initializing SHA %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()  # 获取所有images的id list

        # 截取一部分id，从1到总数*split_ratio范围
        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]

        self.num_samples = len(self.images)
        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        labels = np.array([self.cat_ids[anno['category_id']] for anno in annotations])
        bboxes = np.array([anno['bbox'] for anno in annotations], dtype=np.float32)
        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
        bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image

        scale_map_path=img_path.replace('images','scale_map').replace('jpg','png')
        scale_map=cv2.imread(scale_map_path)

        """1.random resize"""
        # resize img to short len
        img = resize_img_with_short_len(img, short_len=self.short_len)
        h1,w1=img.shape[0],img.shape[1]
        h_rate1 = h1 / height
        w_rate1 = w1 / width
        scale = max(h1, w1) * 1.0

        # resize bbox at the same time
        for i in range(bboxes.shape[0]):
            bboxes[i, 0] = int(bboxes[i, 0] * w_rate1)
            bboxes[i, 2] = int(bboxes[i, 2] * w_rate1)
            bboxes[i, 1] = int(bboxes[i, 1] * h_rate1)
            bboxes[i, 3] = int(bboxes[i, 3] * h_rate1)

        # resize scale map at the same time
        scale_map=cv2.resize(scale_map,(h1,w1),interpolation=cv2.INTER_CUBIC)
        scale_map=scale_map*h_rate1

        # random resize
        if self.split == 'train':
            # choose a randon scale
            random_scale = np.random.choice(self.rand_scales)
            scale = scale * random_scale
            h2 = int(h1 * random_scale)
            w2 = int(w1 * random_scale)

            # resize img
            img = cv2.resize(img, (w2, h2))

            # resize bbox
            bboxes = bboxes * random_scale
            bboxes = bboxes.astype(np.int)

            # resize scale map
            scale_map=cv2.resize(scale_map,(w2,h2),interpolation=cv2.INTER_CUBIC)
            scale_map=scale_map*random_scale

        """2. randon crop"""
        # randon crop img
        if self.split == 'train':
            crop_size = self.img_size['h']
            crop_x1 = np.random.randint(w2 - crop_size)
            crop_y1 = np.random.randint(h2 - crop_size)
            crop_x2 = crop_x1 + crop_size
            crop_y2 = crop_y1 + crop_size
            img = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # crop bbox
            new_box_list = []
            for i in range(bboxes.shape[0]):
                c_x = (bboxes[i][0] + bboxes[i][2]) / 2
                c_y = (bboxes[i][1] + bboxes[i][3]) / 2
                if c_x > crop_x1 and c_x < crop_x2 and c_y > crop_y1 and c_y < crop_y2:
                    x1 = max(bboxes[i][0], crop_x1) - crop_x1
                    y1 = max(bboxes[i][1], crop_y1) - crop_y1
                    x2 = min(bboxes[i][2], crop_x2) - crop_x1
                    y2 = min(bboxes[i][3], crop_y2) - crop_y1
                    new_box_list.append([x1, y1, x2, y2])
            bboxes = np.array(new_box_list)

            if len(new_box_list) == 0:
                bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)

            # crop scale map
            scale_map=scale_map[crop_y1:crop_y2, crop_x1:crop_x2]

        """3. random flip"""
        if self.split == 'train':
            if np.random.random() < 0.5:
                img = img[:, ::-1, :]

                # flip bbox
                h, w, c = img.shape
                new_bboxs = np.zeros(bboxes.shape)
                new_bboxs[:, 0] = w - bboxes[:, 2]
                new_bboxs[:, 1] = bboxes[:, 1]
                new_bboxs[:, 2] = w - bboxes[:, 0]
                new_bboxs[:, 3] = bboxes[:, 3]
                bboxes = new_bboxs.copy()

        """4. color aug"""
        img = img.astype(np.float32) / 255
        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        """5. normalize img"""
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)

        """--------------------generate heatmap and density map ground truth-------------------"""
        bboxes = bboxes // self.down_ratio
        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']),dtype=np.float32)  # heatmap
        density_map=np.zeros((self.fmap_size['h'],self.fmap_size['w']),dtype=np.float32)  # density map gt
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        labels = np.array([0 for _ in range(bboxes.shape[0])])

        # down sample scale map
        scale_map=cv2.resize(scale_map,(scale_map.shape[1]//self.down_ratio,
                                        scale_map.shape[0]//self.down_ratio),cv2.INTER_CUBIC)
        scale_map=scale_map//self.down_ratio

        # generate density map, heatmap, regs ground truth
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                 dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius=int(min(h,w)*self.radius_ratio)
                draw_umich_gaussian(hmap[label], obj_c_int, radius,overlap='max')   # draw heatmap
                draw_umich_gaussian(density_map, obj_c, radius, overlap='add')      # draw density map
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1

        return {'image': img,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id,'scale_map':scale_map,'density_map':density_map}

    def __len__(self):
        return self.num_samples
