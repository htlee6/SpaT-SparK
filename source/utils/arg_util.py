# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys

from tap import Tap

import dist

import argparse


"""
class Args(Tap):
    # environment
    exp_name: str = 'resnet18_chan12'
    exp_dir: str = 'experiments'   # will be created if not exists
    data_path: str = 'imagenet_data_path'
    init_weight: str = ''   # use some checkpoint as model weight initialization; ONLY load model weights
    resume_from: str = ''   # resume the experiment from some checkpoint.pth; load model weights, optimizer states, and last epoch
    
    # SparK hyperparameters
    mask: float = 0.6   # mask ratio, should be in (0, 1)
    mask_ratio_scheduler_type: str = None
    down_rt: int = 32   # downsample ratio (patch size)
    
    # encoder hyperparameters
    model: str = 'resnet18_chan12'
    input_size: int = 288
    sbn: bool = True
    in_channels: int = 12
    
    # data hyperparameters
    bs: int = 256
    dataloader_workers: int = 8
    
    # pre-training hyperparameters
    dp: float = 0.0
    base_lr: float = 2e-4
    wd: float = 0.04
    wde: float = 0.2
    ep: int = 1600
    wp_ep: int = 40
    clip: int = 5.
    opt: str = 'lamb'
    ada: float = 0.
    weight_decay: float = 0.05
    
    # NO NEED TO SPECIFIED; each of these args would be updated in runtime automatically
    lr: float = None
    batch_size_per_gpu: int = 0
    glb_batch_size: int = 0
    densify_norm: str = ''
    device: str = 'cpu'
    local_rank: int = 0
    cmd: str = ' '.join(sys.argv[1:])
    commit_id: str = os.popen(f'git rev-parse HEAD').read().strip() or '[unknown]'
    commit_msg: str = (os.popen(f'git log -1').read().strip().splitlines() or ['[unknown]'])[-1].strip()
    last_loss: float = 0.
    cur_ep: str = ''
    remain_time: str = ''
    finish_time: str = ''
    first_logging: bool = True
    log_txt_name: str = '{args.exp_dir}/pretrain_log.txt'
    tb_lg_dir: str = ''     # tensorboard log directory

    # slurm related
    partition: str = ''
    ngpus: int = 1
    nodes: int = 1
    timeout: int = 5760
    job_dir: str = ''
    comment: str = ''
    mail_user: str = 'h.li2@uu.nl'
    mail_type: str = 'ALL'
    dependency: str = None
    
    def is_convnext(self):
        return 'convnext' in self.model or 'cnx' in self.model

    def is_resnet(self):
        return 'resnet' in self.model
    
    def log_epoch(self):
        if not dist.is_local_master():
            return
        
        if self.first_logging:
            self.first_logging = False
            with open(self.log_txt_name, 'w') as fp:
                json.dump({
                    'name': self.exp_name, 'cmd': self.cmd, 'git_commit_id': self.commit_id, 'git_commit_msg': self.commit_msg,
                    'model': self.model,
                }, fp)
                fp.write('\n\n')
        
        with open(self.log_txt_name, 'a') as fp:
            json.dump({
                'cur_ep': self.cur_ep,
                'last_L': self.last_loss,
                'rema': self.remain_time, 'fini': self.finish_time,
            }, fp)
            fp.write('\n')
"""


def get_args_parser():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--is_pretraining', action='store_true')
    parser.add_argument('--exp_name', default='resnet18_chan12', type=str)
    parser.add_argument('--exp_dir', default='experiments', type=str)
    parser.add_argument('--data_path', default='../train_test_2016-2019_input-length_12_img-ahead_12_rain-threshhold_50.h5', type=str)
    parser.add_argument('--init_weight', default='', type=str)
    parser.add_argument('--resume_from', default='', type=str)

    # SparK hyperparameters
    parser.add_argument('--mask', default=0.6, type=float)
    parser.add_argument('--mask_ratio_scheduler_type', default=None, type=str)
    parser.add_argument('--down_rt', default=32, type=int)
    parser.add_argument('--freeze', default='', type=str)
    parser.add_argument('--transition', default='linear', type=str)

    # encoder hyperparameters
    parser.add_argument('--model', default='resnet18_chan12', type=str)
    parser.add_argument('--input_size', default=288, type=int)
    parser.add_argument('--sbn', default=True, type=bool)
    parser.add_argument('--in_channels', default=12, type=int)

    # data hyperparameters
    parser.add_argument('--bs', default=256, type=int)
    parser.add_argument('--dataloader_workers', default=8, type=int)
    parser.add_argument('--datamode', default='CHW', type=str)
    parser.add_argument('--in_seq_len', default=12, type=int)
    parser.add_argument('--out_seq_len', default=12, type=int)

    # pre-training hyperparameters
    parser.add_argument('--dp', default=0.0, type=float)
    parser.add_argument('--base_lr', default=2e-4, type=float)
    parser.add_argument('--wd', default=0.04, type=float)
    parser.add_argument('--wde', default=0.2, type=float)
    parser.add_argument('--ep', default=1600, type=int)
    parser.add_argument('--wp_ep', default=40, type=int)
    parser.add_argument('--clip', default=5., type=float)
    parser.add_argument('--opt', default='lamb', type=str)
    parser.add_argument('--ada', default=0., type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # NO NEED TO SPECIFIED; each of these args would be updated in runtime automatically
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--batch_size_per_gpu', default=0, type=int)
    parser.add_argument('--glb_batch_size', default=0, type=int)
    parser.add_argument('--densify_norm', default='', type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--cmd', default=' '.join(sys.argv[1:]), type=str)
    parser.add_argument('--commit_id', default=os.popen('git rev-parse HEAD').read().strip() or '[unknown]', type=str)
    parser.add_argument('--commit_msg',
                        default=(os.popen('git log -1').read().strip().splitlines() or ['[unknown]'])[-1].strip(),
                        type=str)
    parser.add_argument('--last_loss', default=0., type=float)
    parser.add_argument('--cur_ep', default='', type=str)
    parser.add_argument('--remain_time', default='', type=str)
    parser.add_argument('--finish_time', default='', type=str)
    parser.add_argument('--first_logging', default=True, type=bool)
    parser.add_argument('--log_txt_name', default='{args.exp_dir}/pretrain_log.txt', type=str)
    parser.add_argument('--tb_lg_dir', default='', type=str)
    parser.add_argument('--Ltrain', default=0., type=float)
    parser.add_argument('--Lval_pixel', default=0., type=float)
    parser.add_argument('--Lval_single_pixel', default=0., type=float)
    parser.add_argument('--CSIval', default={}, type=dict)

    # slurm related

    '''
    parser.add_argument('--partition', default='', type=str)
    parser.add_argument('--ngpus', default=1, type=int)
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--timeout', default=5760, type=int)
    parser.add_argument('--job_dir', default='', type=str)
    parser.add_argument('--comment', default='', type=str)
    parser.add_argument('--mail_user', default='h.li2@uu.nl', type=str)
    parser.add_argument('--mail_type', default='ALL', type=str)
    parser.add_argument('--dependency', default=None, type=str)
    '''

    return parser


def init_dist_and_get_args(args):
    from utils import misc
    
    # initialize
    e = os.path.abspath(args.exp_dir)
    d, e = os.path.dirname(e), os.path.basename(e)
    e = ''.join(ch if (ch.isalnum() or ch == '-') else '_' for ch in e)
    args.exp_dir = os.path.join(d, e)
    
    os.makedirs(args.exp_dir, exist_ok=True)
    args.log_txt_name = os.path.join(args.output_dir if hasattr(args, 'output_dir') else args.exp_dir, 'pretrain_log.txt' if args.is_pretraining else 'finetune_log.txt')
    args.tb_lg_dir = args.tb_lg_dir or os.path.join(args.log_dir if hasattr(args, 'output_dir') else args.exp_dir, 'tensorboard_log')
    try:
        os.makedirs(args.tb_lg_dir, exist_ok=True)
    except:
        pass
    
    misc.init_distributed_environ(exp_dir=args.exp_dir)
    
    # update args
    if not dist.initialized():
        args.sbn = False
    args.first_logging = True
    args.device = dist.get_device()
    args.batch_size_per_gpu = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size_per_gpu * dist.get_world_size()
    
    if is_resnet(args):
        args.ada = args.ada or 0.95
        args.densify_norm = 'bn'
    
    if is_convnext(args):
        args.ada = args.ada or 0.999
        args.densify_norm = 'ln'
    
    args.opt = args.opt.lower()
    args.lr = args.base_lr * args.glb_batch_size / 256
    args.wde = args.wde or args.wd
    
    return args


def is_resnet(args):
    return 'resnet' in args.model


def is_convnext(args):
    return 'convnext' in args.model or 'cnx' in args.model


def log_epoch(args):
    if not dist.is_local_master():
        return

    if args.first_logging:
        args.first_logging = False
        with open(args.log_txt_name, 'w') as fp:
            json.dump({
                'name': args.exp_name, 'cmd': args.cmd, 'git_commit_id': args.commit_id,
                'git_commit_msg': args.commit_msg,
                'model': args.model,
            }, fp)
            fp.write('\n\n')

    with open(args.log_txt_name, 'a') as fp:
        if args.is_pretraining:
            json.dump({
                'cur_ep': args.cur_ep,
                'cur_lr': args.lr,
                'last_L': args.last_loss,
                'rema': args.remain_time, 'fini': args.finish_time,
            }, fp)
            fp.write('\n')
        else:
            json.dump({
                'cur_ep': args.cur_ep,
                'cur_lr': args.lr,
                'Ltrain': args.Ltrain,
                'Lval': args.last_loss,
                'Lval_pixel': args.Lval_pixel,
                'Lval_single_pixel': args.Lval_single_pixel,
                'CSIval': args.CSIval,
                'rema': args.remain_time, 'fini': args.finish_time,
            }, fp)
            fp.write('\n')