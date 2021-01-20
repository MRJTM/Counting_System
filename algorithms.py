"""
@File       : algorithms.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2020/12/2
@Desc       : APIs of the Counting System
"""

import time
import torch
import sys

sys.path.append('..')
from utils.visualize import *
from utils.image import resize_img_with_short_len, img_to_tensor
from utils.model import predict_bbox_and_density


def draw_boxs(img, boxs, width=3, color=(0, 0, 255)):
    box_img = copy.deepcopy(img)
    for i in range(boxs.shape[0]):
        x1, y1, x2, y2 = boxs[i][:4]
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(box_img, p1, p2, color, width)
    return box_img


def resize_gt(gt, h, w):
    resized_gt = cv2.resize(gt, (w, h))
    gt_num_before_resize = np.sum(gt)
    gt_num_after_resize = np.sum(resized_gt)
    if gt_num_before_resize == 0 or gt_num_after_resize == 0:
        gt_change_rate = 1
    else:
        gt_change_rate = gt_num_before_resize / gt_num_after_resize
    resized_gt = resized_gt * gt_change_rate
    return resized_gt


def test_one_image(img, model, down_ratio=2, short_len=512, score_threshold=0.3, device=''):
    time1 = time.time()
    print("enter_test_one_image")
    img = resize_img_with_short_len(img, short_len=short_len)
    height, width = img.shape[0], img.shape[1]
    img_raw = copy.deepcopy(img)
    img1 = copy.deepcopy(img)

    """turn img to tensor"""
    img1 = img_to_tensor(img1)
    if torch.cuda.is_available() and device == 'gpu':
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    img1 = img1.to(dev)

    """model forward"""
    time2 = time.time()
    threshold = score_threshold
    ratio = [down_ratio, down_ratio]
    preds = predict_bbox_and_density(model, img1, cfg=None, threshold=threshold, down_ratio=ratio, device=device)
    print("get preds")
    pred_bbox = preds[0]
    pred_hmap = preds[1]
    pred_smap = preds[2]
    pred_density = preds[3]
    print("model_forward_time:", time.time() - time2)

    # visualize bbox img
    box_img = draw_boxs(img_raw, pred_bbox[:, :4], width=2, color=(0, 0, 255))
    box_img = cv2.cvtColor(box_img, cv2.COLOR_RGB2BGR)

    # visualize heatmap
    pred_hmap = torch.sigmoid(pred_hmap)
    pred_hmap = pred_hmap.detach().cpu().numpy().reshape(pred_hmap.shape[2], pred_hmap.shape[3])
    pred_heatmap = generate_heatmap(pred_hmap, rate=0)
    pred_heatmap = cv2.resize(pred_heatmap, (width, height))
    pred_heatmap = cv2.cvtColor(pred_heatmap, cv2.COLOR_RGB2BGR)

    # visualize scale map
    pred_smap = pred_smap.detach().cpu().numpy().reshape(pred_smap.shape[2], pred_smap.shape[3])
    pred_smap = pred_smap * down_ratio
    pred_smap = cv2.resize(pred_smap, (width, height))
    pred_smap_heatmap = generate_heatmap(pred_smap, rate=0)
    pred_smap_heatmap = cv2.cvtColor(pred_smap_heatmap, cv2.COLOR_RGB2BGR)

    # visualize density map
    pred_density = pred_density.detach().cpu().numpy().reshape(pred_density.shape[2],
                                                               pred_density.shape[3])
    pred_density = resize_gt(pred_density, height, width)
    pred_density_heatmap = generate_heatmap(pred_density, rate=0)
    pred_density_heatmap = cv2.cvtColor(pred_density_heatmap, cv2.COLOR_RGB2BGR)
    print("test_one_img cost time:", time.time() - time1)

    return pred_bbox, pred_hmap, pred_smap, pred_density, \
           box_img, pred_heatmap, pred_smap_heatmap, pred_density_heatmap


# combine density map and detections
def combine_two_result(img, pred_density, pred_bboxs, pred_smap, scale_threshold, short_len):
    img = resize_img_with_short_len(img, short_len=short_len)
    img_raw = copy.deepcopy(img)
    smap_threshold = scale_threshold

    """get the density map where scale value < scale_threshold"""
    pred_density_part = pred_density.copy()
    pred_density_part[np.where(pred_smap >= scale_threshold)] = 0
    density_part_num = np.sum(pred_density_part)

    pred_density_part_heatmap = generate_heatmap(pred_density_part, rate=0)
    pred_density_part_heatmap[np.where(pred_density_part_heatmap == \
                                       pred_density_part_heatmap[50, 50, :])] = 0

    """get bbox where scale value > scale_threshold"""
    new_box_list = []
    for i in range(pred_bboxs.shape[0]):
        x1, y1, x2, y2, score = pred_bboxs[i]
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        if pred_smap[center_y, center_x] >= smap_threshold:
            new_box_list.append([x1, y1, x2, y2])
    bbox_part_num = len(new_box_list)
    bbox_part = np.array(new_box_list)
    bbox_part_img = draw_boxs(img_raw, bbox_part, width=2, color=(0, 0, 255))

    """combine density part and bbox part"""
    alpha = 0.9
    combined_img = cv2.addWeighted(pred_density_part_heatmap, alpha, bbox_part_img,
                                   1 - alpha, 0)
    combined_img = combined_img
    combined_img[np.where(pred_density_part_heatmap == 0)] = bbox_part_img[np.where(pred_density_part_heatmap == 0)]
    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

    pred_num = density_part_num + bbox_part_num
    return combined_img, pred_num


def draw_boxs_with_id(img, bbox_dict, width, h_rate=1,w_rate=1,show_gt=False, color=None, color_dict=None):
    box_img = copy.deepcopy(img)
    for box_id in bbox_dict.keys():
        x1, y1, x2, y2 = bbox_dict[box_id]
        x1 = int(x1*w_rate)
        y1 = int(y1*h_rate)
        x2 = int(x2*w_rate)
        y2 = int(y2*h_rate)
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        if not color:
            color = color_dict[box_id]
        cv2.rectangle(box_img, p1, p2, color, width)
        if show_gt:
            cv2.putText(box_img, "{:d}".format(int(box_id)), p1, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
    return box_img


def IOU(bbox1, bbox2):
    x11, y11, x12, y12 = bbox1
    x21, y21, x22, y22 = bbox2
    xmin = max(x11, x21)
    xmax = min(x12, x22)
    ymin = max(y11, y21)
    ymax = min(y12, y22)
    if xmax > xmin and ymax > ymin:
        inter = (xmax - xmin) * (ymax - ymin)
        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        iou = inter / (area1 + area2 - inter)
    else:
        iou = 0
    return iou

# if cross the boarder of the ROI
def cross_line(last_bbox, cur_bbox, ROI):
    r_x1, r_y1, r_x2, r_y2 = ROI
    last_x1, last_y1, last_x2, last_y2 = last_bbox
    cur_x1, cur_y1, cur_x2, cur_y2 = cur_bbox
    last_cx = (last_x1 + last_x2) / 2
    last_cy = (last_y1 + last_y2) / 2
    cur_cx = (cur_x1 + cur_x2) / 2
    cur_cy = (cur_y1 + cur_y2) / 2
    cross = False
    if last_cx < r_x1 and cur_cx > r_x1 and cur_cy > r_y1 and cur_cy < r_y2:
        cross = True
    elif last_cx > r_x2 and cur_cx < r_x2 and cur_cy > r_y1 and cur_cy < r_y2:
        cross = True
    elif last_cy < r_y1 and cur_cy > r_y1 and cur_cx > r_x1 and cur_cx < r_x2:
        cross = True
    elif last_cy > r_y2 and cur_cy < r_y2 and cur_cx > r_x1 and cur_cx < r_x2:
        cross = True

    return cross


def track_with_det_and_KCF(cur_frame, cur_bbox_dict, last_bbox_dict, tracker_dict, max_id,
                           iou_threshold=0.5, ROI=[]):
    res_bbox_dict = {}
    res_tracker_dict = {}
    flow_count = 0

    # match track result with cur_bbox_dict
    for person_id in tracker_dict.keys():
        ok, track_bbox = tracker_dict[person_id].update(cur_frame)
        track_bbox1 = [track_bbox[0], track_bbox[1],
                       track_bbox[0] + track_bbox[2],
                       track_bbox[1] + track_bbox[3]]
        if ok:
            delete_key = -1
            for key in cur_bbox_dict.keys():
                cur_bbox = cur_bbox_dict[key]
                # if match
                if IOU(track_bbox1, cur_bbox) > iou_threshold:
                    res_bbox_dict[person_id] = cur_bbox
                    res_tracker_dict[person_id] = tracker_dict[person_id]
                    delete_key = key

                    # if cross line
                    last_bbox = last_bbox_dict[person_id]
                    if cross_line(last_bbox, cur_bbox, ROI):
                        flow_count += 1

                    break

            if delete_key > -1:
                del cur_bbox_dict[delete_key]
            else:
                cur_bbox = track_bbox1
                res_bbox_dict[person_id] = cur_bbox
                res_tracker_dict[person_id] = tracker_dict[person_id]

                # if cross line
                last_bbox = last_bbox_dict[person_id]
                if cross_line(last_bbox, cur_bbox, ROI):
                    flow_count += 1

    # give a new id to unmatched cur_detection
    new_id = max_id
    for key in cur_bbox_dict.keys():
        new_id += 1
        cur_bbox = cur_bbox_dict[key]
        res_bbox_dict[new_id] = cur_bbox
        # give each bbox a tracker
        tracker = cv2.TrackerKCF_create()
        init_box = (cur_bbox[0], cur_bbox[1],
                    cur_bbox[2] - cur_bbox[0],
                    cur_bbox[3] - cur_bbox[1])
        tracker.init(cur_frame, init_box)
        res_tracker_dict[new_id] = tracker

    return res_bbox_dict, res_tracker_dict, new_id, flow_count


def track_with_only_KCF(cur_frame, last_bbox_dict, tracker_dict, ROI=[]):
    res_bbox_dict = {}
    res_tracker_dict = {}
    flow_count = 0

    # match trackers with last bbox
    for person_id in tracker_dict.keys():
        ok, track_bbox = tracker_dict[person_id].update(cur_frame)
        track_bbox1 = [track_bbox[0], track_bbox[1],
                       track_bbox[0] + track_bbox[2],
                       track_bbox[1] + track_bbox[3]]
        if ok:
            cur_bbox = track_bbox1
            res_bbox_dict[person_id] = cur_bbox
            res_tracker_dict[person_id] = tracker_dict[person_id]

            # if cross line
            last_bbox = last_bbox_dict[person_id]
            if cross_line(last_bbox, cur_bbox, ROI):
                flow_count += 1

    return res_bbox_dict, res_tracker_dict, flow_count


def track_with_only_det(cur_bbox_dict, last_bbox_dict, iou_threshold=0.5, max_id=0, ROI=[]):
    res_bbox_dict = {}
    flow_count = 0

    # match bboxs
    for person_id in last_bbox_dict.keys():
        last_bbox = last_bbox_dict[person_id]
        delete_key = -1
        for key in cur_bbox_dict.keys():
            cur_bbox = cur_bbox_dict[key]
            # if match
            if IOU(last_bbox, cur_bbox) > iou_threshold:
                res_bbox_dict[person_id] = cur_bbox
                delete_key = key

                # if cross line
                last_bbox = last_bbox_dict[person_id]
                if cross_line(last_bbox, cur_bbox, ROI):
                    flow_count += 1

                break

        if delete_key > -1:
            del cur_bbox_dict[delete_key]

    # give a new id to unmatched bbox
    new_id = max_id
    for key in cur_bbox_dict.keys():
        new_id += 1
        cur_bbox = cur_bbox_dict[key]
        res_bbox_dict[new_id] = cur_bbox

    return res_bbox_dict, new_id, flow_count

# pack bbox array and ids to a bbox_dict
def generate_bbox_dict_from_bbox(bbox, gap, height, width):
    bbox_dict = {}
    for i in range(bbox.shape[0]):
        x1 = int(max(bbox[i][0] - gap, 0))
        y1 = int(max(bbox[i][1] - gap, 0))
        x2 = int(min(bbox[i][2] + gap, width - 1))
        y2 = int(min(bbox[i][3] + gap, height - 1))
        if x2 > x1 and y2 > y1:
            bbox_dict[i] = [x1, y1, x2, y2]

    return bbox_dict

def get_bbox_by_model(model, img, threshold, down_ratio, device):
    # det heads
    img1 = img.copy()
    img1 = img_to_tensor(img1)
    if torch.cuda.is_available() and device == 'gpu':
        dev = torch.device('cuda:0')
    else:
        dev = torch.device('cpu')
    img1 = img1.to(dev)
    preds = predict_bbox_and_density(model, img1, cfg=None, threshold=threshold,
                                     down_ratio=down_ratio, device=device)
    return preds

