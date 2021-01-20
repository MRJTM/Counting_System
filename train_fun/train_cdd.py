"""
@File       : train_ccd.py
@Author     : Zhijie Cao
@Email      : dearzhijie@gmail.com
@Date       : 2020/9/23
@Desc       : train CDDNet
"""

import time
import torch
from utils.losses import _neg_loss, _reg_loss,scale_map_loss
from utils.utils import _tranpose_and_gather_feature

hmap_loss_dict={
    'focal':_neg_loss,
    'mse':torch.nn.MSELoss(reduction='sum')
}

def train_CDDNet(epoch,model,train_loader,cfg,optimizer,summary_writer):
    print('\n Epoch: %d' % epoch)
    model.train()
    tic = time.perf_counter()
    for batch_idx, batch in enumerate(train_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

        outputs = model(batch['image'])
        hmap, regs, smap, density = zip(*outputs)

        regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]

        """----heatmap loss----"""
        hmap_loss_names = cfg.hmap_loss_names.split(',')
        hmap_loss = 0
        for hmap_loss_name in hmap_loss_names:
            if hmap_loss_name=='mse':
                loss_func = hmap_loss_dict[hmap_loss_name].to(cfg.device)
                for pred_hmap in hmap:
                    pred_hmap = torch.sigmoid(pred_hmap)
                    hmap_loss += 0.001*loss_func(pred_hmap, batch['hmap'])
            else:
                hmap_loss += hmap_loss_dict[hmap_loss_name](hmap, batch['hmap'])

        """----scale map loss----"""
        pred_smap=smap[0].squeeze()
        gt_smap = batch['scale_map'][:, :, :, 0].type(torch.FloatTensor).to(device=cfg.device, non_blocking=True)
        smap_loss = scale_map_loss(pred_smap, gt_smap)

        """----reg loss----"""
        reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])

        """----density map loss----"""
        pred_density_map=density[0].squeeze()
        mse_loss=torch.nn.MSELoss(reduction='sum').to(cfg.device)
        density_loss=mse_loss(pred_density_map,batch['density_map'])

        """----total loss----"""
        loss = hmap_loss + 1 * reg_loss + 0.1 * smap_loss+0.2*density_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """--------log loss-----------"""
        if batch_idx % cfg.log_interval == 0:
            duration = time.perf_counter() - tic
            tic = time.perf_counter()
            print('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                  ' hmap_loss= %.5f reg_loss= %.5f smap_loss= %.5f den_loss= %.5f' %
                  (hmap_loss.item(), reg_loss.item(), smap_loss.item(), density_loss.item()) +
                  ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

            step = len(train_loader) * epoch + batch_idx
            summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
            summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
            summary_writer.add_scalar('smap_loss', smap_loss.item(), step)
            summary_writer.add_scalar('den_loss',density_loss.item(),step)
    return model