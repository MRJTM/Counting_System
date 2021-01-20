
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
  hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
  keep = (hmax == heat).float()
  # print("heat.shape:",heat.shape)
  return heat * keep


def _topk(scores, K=40):
  batch, cat, height, width = scores.size()

  # 每个类别选出topK
  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()

  # 选出所有类别中的topk
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100):
  batch, cat, height, width = hmap.shape
  hmap=torch.sigmoid(hmap)
  # print("hmap.shape:",hmap.shape) # [n,num_class,H,W]
  # print("regs.shape:",regs.shape) # [n,2,H,W]
  # print("w_h_.shape:",w_h_.shape) # [n,2,H,W]

  # if flip test_fun
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
    regs = regs[0:1]

  batch = 1

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  w_h_ = _tranpose_and_gather_feature(w_h_, inds)
  w_h_ = w_h_.view(batch, K, 2)

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections

def ctdet_decode_with_scale_map(hmap,regs,smap,cfg,K=100,device='gpu'):
  batch, cat, height, width = hmap.shape
  hmap = torch.sigmoid(hmap)
  # print("hmap.shape:",hmap.shape) # [n,num_class,H,W]
  # print("regs.shape:",regs.shape) # [n,2,H,W]
  # print("w_h_.shape:",w_h_.shape) # [n,2,H,W]

  # if flip test_fun
  if batch > 1:
    hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
    regs = regs[0:1]
    smap = (smap[0:1] + flip_tensor(smap[1:2])) / 2

  batch = 1

  hmap = _nms(hmap)  # perform nms on heatmaps

  scores, inds, clses, ys, xs = _topk(hmap, K=K)

  regs = _tranpose_and_gather_feature(regs, inds)
  regs = regs.view(batch, K, 2)
  xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
  ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

  if cfg:
    w_h_=torch.zeros((batch,K,2),dtype=torch.float).to(cfg.device)
  else:
    if device=='gpu':
      dev=torch.device('cuda:0')
    else:
      dev=torch.device('cpu')
    w_h_ = torch.zeros((batch, K, 2), dtype=torch.float).to(dev)

  smap=smap.squeeze()

  for i in range(K):
    x=int(xs[0,i,0].detach().cpu().numpy())
    y=int(ys[0,i,0].detach().cpu().numpy())
    x=min(x,smap.shape[1]-1)
    y=min(y,smap.shape[0]-1)
    w=smap[y,x]
    h=smap[y,x]
    w_h_[0,i,0]=w
    w_h_[0,i,1]=h

  clses = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)
  # xs=xs.cpu()
  # ys=ys.cpu()
  # w_h_=w_h_.cpu()
  bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                      ys - w_h_[..., 1:2] / 2,
                      xs + w_h_[..., 0:1] / 2,
                      ys + w_h_[..., 1:2] / 2], dim=2)
  detections = torch.cat([bboxes, scores, clses], dim=2)
  return detections


