import torch
import torch.nn as nn
import torch.nn.functional as F


def _neg_loss_slow(preds, targets):
  pos_inds = targets == 1  # todo targets > 1-epsilon ?
  neg_inds = targets < 1  # todo targets < 1-epsilon ?

  neg_weights = torch.pow(1 - targets[neg_inds], 4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(preds, targets):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      preds (B x c x h x w)
      gt_regr (B x c x h x w)
  '''
  pos_inds = targets.eq(1).float()  # 等于
  # pos_inds=targets.ge(1).float()      # 大于等于
  neg_inds = targets.lt(1).float()    # 小于

  neg_weights = torch.pow(1 - targets, 4)
  # max_diff=torch.max(torch.max(targets)-targets)
  # neg_weights=torch.pow((torch.max(targets)-targets)/max_diff,4)

  loss = 0
  for pred in preds:
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) / num_pos
  return loss / len(preds)



def _reg_loss(regs, gt_regs, mask):
  mask = mask[:, :, None].expand_as(gt_regs).float()
  loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
  return loss / len(regs)

def scale_map_loss(pd_smap,gt_smap):
  n,h,w=gt_smap.shape
  loss=torch.sum(torch.abs(pd_smap-gt_smap))/(n*h*w)
  return loss

def _sg_loss(real_feature,syn_feature):
  mse_loss=nn.MSELoss(reduction='sum')
  loss=mse_loss(real_feature,syn_feature)
  return loss

# def scale_map_loss(pd_smap,gt_smap):
#   n,h,w=gt_smap.shape
#   # L1 loss
#   L1_loss=torch.sum(torch.abs(pd_smap-gt_smap))/(n*h*w)
#   # std loss
#   std_loss=0
#   for i in range(n):
#     for j in range(h):
#       pred_line=pd_smap[i,j,:]
#       gt_line=gt_smap[i,j,:]
#       pred_mean=torch.mean(pred_line)
#       gt_mean=torch.mean(gt_line)
#       pred_std=torch.std(pred_line)
#       std_loss+=torch.abs(pred_mean-gt_mean)
#       std_loss+=pred_std
#   std_loss=std_loss/(n*h)
#   loss=L1_loss+std_loss
#   return loss