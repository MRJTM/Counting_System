"""
@File       : visualize.py
@Author     : caozhijie
@Email      : caozhijie1@sensetime.com
@Date       : 2020/3/23
@Desc       : functions for visualization on jupyter, like draw point, draw box, generate heatmap...
"""

import cv2
import numpy as np
import copy

def draw_point_gt(img,point_gt,width=3):
    """
    draw_point_gt is to show point gt on the img with read circles

    Args:
        img: cv2 format img
        point_gt: numpy matrix, [[x1,y1],[x2,y2],.....]
        width: the size of point

    Returns:
        point_img: img with drawed points

    """
    point_img=copy.deepcopy(img)

    for i in range(point_gt.shape[0]):
        x=int(point_gt[i,0])
        y=int(point_gt[i,1])
        point_img=cv2.circle(point_img,center=(x,y),radius=width//2+1,color=(0,0,255),thickness=width)

    return point_img

def generate_heatmap(density_map,rate=0,color_type=2):
    """
    generate_heatmap is to generate heatmap from density map

    Args:
        density_map: numpy array
        rate: if rate=0, we will generate heatmap by dividing density map by it's max value;
              if rate>0, we will generate heatmap by directly multiplying density map by rate.
        color_type: int, to choose the color type of heatmap, in [1,2,3....8]

    Returns:
        heatmap: generated heatmap, a 3 channel color image of cv2

    """

    if rate > 0:
        heatmap = np.abs(density_map) * rate
    else:
        max_density_value=np.max(np.abs(density_map))
        if max_density_value==0:
            max_density_value=1
        heatmap = np.abs(density_map) / max_density_value * 250
    heatmap = cv2.applyColorMap(np.uint8(heatmap), color_type)
    return heatmap

def draw_scale(img,scale,width=2,head_enlarge_rate=1,fix_head_size=0):
    """
    draw_scale is to draw scale on the img, each point of scale will draw a rectangle on img.
    
    Args:
        img: cv2 like img
        scale: numpy array, [[x1,y1,s1],[x2,y2,s2]...[xn,yn,sn]]
        width: the linewidth of the rectangle
        head_enlarge_rate: if fix_head_size=0, the size of rectangle is s*head_enlarge_rate;
                           if fix_head_size>0, the size of rectangle equals fix_head_size.
        fix_head_size: see in head_enlarge_rate
    
    Returns:
        scale_img: the img with rectangles
    
    """
    scale_img = copy.deepcopy(img)
    for i in range(scale.shape[0]):
        if fix_head_size > 0:
            head_size = fix_head_size
        else:
            head_size = scale[i][2] * head_enlarge_rate
        p1 = (int(round(scale[i][0] - head_size / 2)),
              int(round(scale[i][1] - head_size / 2)))
        p2 = (int(round(scale[i][0] + head_size / 2)),
              int(round(scale[i][1] + head_size / 2)))
        cv2.rectangle(scale_img, p1, p2, (0, 0, 255), width)

    return scale_img

def draw_boxs(img,boxs,width=3,color=(0,0,255)):
    """
    draw_boxs is to draw bounding boxs on the img

    Args:
        img: cv2 like image
        boxs: numpy array, [[x1,y1,x2,y2],,,,]
        width: the width of bounding box

    Returns:
        box_img: the img with boxs

    """
    box_img = copy.deepcopy(img)
    for i in range(boxs.shape[0]):
        # x1,y1,x2,y2=boxs[i]
        x1 = boxs[i][0]
        y1 = boxs[i][1]
        x2 = boxs[i][2]
        y2 = boxs[i][3]
        p1 = (int(round(x1)),int(round(y1)))
        p2 = (int(round(x2)),int(round(y2)))
        cv2.rectangle(box_img, p1, p2, color, width)

    return box_img


def visualize_id_box_point(img,person_dict={},point_size=3,box_width=3,font_size=1):
    """
    visualize_id_box_point is to show id, head point, box of the MOT result,
    different id will be presented with different color

    Args:
        img: the cv2 type image, is a numpy array
        person_dict: a dict, which is :
                    {
                        '1':{'point':[x,y],'bbox':[x1,y1,x2,y2]},
                        '2':{'point':[x,y],'bbox':[x1,y1,x2,y2]},
                        ...
                    }
        point_size: the size of head point, point_size=0 means not show the head point
        box_width: the linewidth of head_should_box, if 0, means not show the box
        font_size: the font size of id, if 0 ,means not show the id
    Returns:
        result_img: the img with id, point, box

    """
    result_img=copy.deepcopy(img)
    img_w=img.shape[1]
    img_h=img.shape[0]
    num_point=0
    num_box=0
    for id in person_dict.keys():
        [hx,hy]=person_dict[id]['point']
        [x1,y1,x2,y2]=person_dict[id]['bbox']

        # determine the color according to id
        B = 32 * (int(id) // 64 + 1) - 1
        G = 32 * ((int(id) % 64) // 8 + 1) - 1
        R = 32 * ((int(id) % 8) + 1) - 1

        # draw point
        if hx>=0 and hx<img_w and hy>=0 and hy<img_h and point_size>0:
            center=(int(hx),int(hy))
            result_img = cv2.circle(result_img, center=center,
                                    radius=point_size // 2 + 2,
                                    color=(B,G,R), thickness=point_size)
            num_point+=1
        else:
            print("point [{}]:({},{}) missed".format(id,hx,hy))

        # draw box
        if x1>=0 and y1>=0 and x2<=img_w and y2<=img_h and box_width>0:
            p1=(int(x1),int(y1))
            p2=(int(x2),int(y2))
            result_img=cv2.rectangle(result_img, p1, p2, (B,G,R), box_width)
            num_box+=1
        else:
            print("box [{}]:({},{},{},{}) missed".format(id,x1,y1,x2,y2))

        # draw id
        if font_size>0:
            p_text=(int(x1),int(y1))
            result_img=cv2.putText(result_img, "{}".format(id), p_text,
                                   cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size,
                                   color=(B,G,R), thickness=font_size*3)
    num_id=len(list(person_dict.keys()))
    print("num_id:",)
    print("num_point:",num_point)
    print("num_box:",num_box)

    return result_img,num_id,num_box,num_point


def get_heatmap_rate(pred, gt):
    max_pred = np.max(pred)
    max_gt = np.max(gt)
    if max_pred == 0:
        pred_rate = 1
    else:
        pred_rate = 254 / max_pred

    if max_gt == 0:
        gt_rate = 1
    else:
        gt_rate = 254 / max_gt

    rate = min(pred_rate, gt_rate)

    return rate
