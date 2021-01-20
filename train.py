import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

sys.path.append(os.getcwd())
from datasets.sha import SHA

from nets.cddnet import CDDNet3
from utils.utils import load_model
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from train_fun.train_cdd import train_CDDNet
from test_fun.val_cdd import val_cdd
from utils.json_utils import *

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet45')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--dataset_name', type=str, default='SHA',)
parser.add_argument('--model_name', type=str, default='CDDNet3')
parser.add_argument('--backbone', type=str, default='vgg16_bn')
parser.add_argument('--model_save_name', type=str, default='saved model name')
parser.add_argument('--pretrain_model_name', type=str, default='pretrain model name')
parser.add_argument('--train_from_scratch', action='store_true')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--train_split_ratio', type=float, default=1.0)
parser.add_argument('--val_split_ratio', type=float, default=1.0)
parser.add_argument('--radius_ratio', type=float, default=0.2)
parser.add_argument('--hmap_loss_names', type=str, default='focal', help='focal,mse')


parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

"""-----------------------------config path-----------------------------"""
os.chdir('./')
cfg.log_dir = os.path.join('./', 'logs', cfg.model_save_name)
cfg.ckpt_dir = os.path.join('./', 'trained_models', cfg.model_save_name)
pretrain_dir = os.path.join('./', 'trained_models', cfg.pretrain_model_name)
pretrain_model_info_path = os.path.join(pretrain_dir, 'info.json')
model_info_path = os.path.join(cfg.ckpt_dir, 'info.json')
cfg.checkpoint_path = os.path.join(pretrain_dir, 'checkpoint.t7')
print("checkpoint_path:", cfg.checkpoint_path)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]

dataset_dict = {
    'SHA': {'train': SHA, 'num_class': 1},
}

down_ratio_dict = {'cddnet3':2}

# config train function
train = train_CDDNet

"""------------main-----------"""
def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
    print(cfg)

    """config GPU"""
    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda')

    """---------------------------------config dataset------------------------------"""
    print('\nSetting up data...')
    Dataset = dataset_dict[cfg.dataset_name]['train']
    train_dataset = Dataset(cfg.data_dir, 'train', split_ratio=cfg.train_split_ratio,img_size=cfg.img_size,
                            down_ratio=down_ratio_dict[cfg.model_name],radius_ratio=cfg.radius_ratio)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=num_gpus,
                                                                    rank=cfg.local_rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size // num_gpus
                                               if cfg.dist else cfg.batch_size,
                                               shuffle=not cfg.dist,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               sampler=train_sampler if cfg.dist else None)

    """--------------------------------config model-------------------------------"""
    print('\nCreating model...')
    print("using model:", cfg.model_name)
    num_classes = dataset_dict[cfg.dataset_name]['num_class']
    if 'cddnet3' == cfg.model_name:
        model = CDDNet3(num_classes=num_classes,backbone=cfg.backbone,load_pretrain=True)
    else:
        raise NotImplementedError

    # load pretrained weight
    if (not cfg.train_from_scratch) and os.path.isfile(cfg.checkpoint_path):
        print("load pretrain model from:{}".format(cfg.checkpoint_path))
        model = load_model(model, cfg.checkpoint_path)
        model_info = read_json_model_info(pretrain_model_info_path)
    else:
        model_info = {
            'best_MAE': 1000,
            'best_epoch': 0,
            'checkpoint_MAE': 1000,
            'checkpoint_epoch': 0
        }
    print("\n[model_info]:")
    for key in model_info.keys():
        print("{}:{}".format(key,model_info[key]))

    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[cfg.local_rank, ],
                                                    output_device=cfg.local_rank)
    else:
        model = nn.DataParallel(model).to(cfg.device)

    """config optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

    """-----------------------training----------------------"""
    print('\n[Starting training]...')
    for epoch in range(model_info['checkpoint_epoch'] + 1, cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        # train model
        model = train(epoch, model, train_loader, cfg, optimizer, summary_writer)

        # val model
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            test_image_folder_path = os.path.join(cfg.data_dir, 'test','images')
            cur_MAE = val_cdd(epoch, model, test_image_folder_path, cfg, summary_writer,
                              down_ratio=down_ratio_dict[cfg.model_name],
                              short_len=640, score_threshold=0.2)

            if cur_MAE < model_info['best_MAE']:
                model_info['best_MAE'] = cur_MAE
                model_info['best_epoch'] = epoch
                write_json_model_info(model_info_path, model_info)
                print(saver.save(model.module.state_dict(), 'best'))

            print("\n----[Compare]----")
            print("[cur]:  cur_MAE={:.3f},cur_epoch={:d}".format(cur_MAE, epoch))
            print("[best]: best_MAE={:.3f},best_epoch={:d}".format(model_info['best_MAE'],
                                                                   model_info['best_epoch']))

            # save checkpoint
            model_info['checkpoint_MAE'] = cur_MAE
            model_info['checkpoint_epoch'] = epoch
            write_json_model_info(model_info_path, model_info)
            print(saver.save(model.module.state_dict(), 'checkpoint'))

        lr_scheduler.step(epoch)
    summary_writer.close()


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()
