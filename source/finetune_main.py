# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import math
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.functional import mse_loss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics import CriticalSuccessIndex, ConfusionMatrix
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from spark import SparK
from spark_ft import SparKFinetune
from utils import arg_util, misc, lamb, mask_ratio_scheduler
from utils.imagenet import build_dataset_to_pretrain
from utils.lr_control import lr_wd_annealing, get_param_groups
from dataset_knmi import precipitation_maps_oversampled_h5, precipitation_maps_oversampled_TCHW
import torchvision.transforms as T

from utils.arg_util import log_epoch
import neptune


class LocalDDP(torch.nn.Module):
    def __init__(self, module):
        super(LocalDDP, self).__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_finetune(args):
    run = neptune.init_run(
        project='htlee6space/sslnowcasting',
        name=args.exp_name,
        tags=[
            'SparK',
            'finetune' if args.init_weight != "" or args.resume_from != "" else 'unpretrained',
            args.transition,
            args.freeze+'-freeze'
        ]
    )

    args = arg_util.init_dist_and_get_args(args)
    print(f'initial args:\n{str(args)}')
    log_epoch(args)

    # add information to neptune from args, args is a namespce object
    # for every key in args, add it to neptune
    for key in vars(args):
        run[key] = getattr(args, key)
    
    # build data
    print(f'[build data for pre-training] ...\n')
    # dataset_train = build_dataset_to_pretrain(args.data_path, args.input_size)
    transform = T.Compose([
        T.RandomCrop(224)
    ])
    if args.in_channels == 1 and args.datamode == 'CHW':
        dataset_train = precipitation_maps_oversampled_h5("../data/train_test_2016-2019_input-length_1_img-ahead_1_rain-threshhold_50.h5", 1, 1, train=True, transform=None)
        dataset_val = precipitation_maps_oversampled_h5("../data/train_test_2016-2019_input-length_1_img-ahead_1_rain-threshhold_50.h5", 1, 1, train=False, transform=None)
    elif args.in_channels == 12 and args.datamode == 'CHW':
        dataset_train = precipitation_maps_oversampled_h5(args.data_path, 12, 12, train=True, transform=None)
        dataset_val = precipitation_maps_oversampled_h5(args.data_path, 12, 12, train=False, transform=None)
    elif args.in_channels == 1 and args.datamode == 'TCHW':
        dataset_train = precipitation_maps_oversampled_TCHW(args.data_path, args.in_seq_len, args.out_seq_len, train=True, transform=None)
        dataset_val = precipitation_maps_oversampled_TCHW(args.data_path, args.in_seq_len, args.out_seq_len, train=False, transform=None)

    else:
        raise ValueError(f'Choose valid in_channels and datamode combination. Valid combinations are: (1, CHW), (12, CHW), (1, TCHW)')
    data_loader_train = DataLoader(
        dataset=dataset_train, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size,
            shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    data_loader_val = DataLoader(
        dataset=dataset_val, num_workers=args.dataloader_workers, pin_memory=True,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_val), glb_batch_size=args.glb_batch_size,
            shuffle=False, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
        ), worker_init_fn=worker_init_fn
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    itrt_val, iters_val = iter(data_loader_val), len(data_loader_val)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    # build encoder and decoder
    if args.datamode == 'CHW':
        enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False, in_chans=args.in_channels)
        dec = LightDecoder(enc.downsample_raito, sbn=args.sbn, out_chan=args.in_channels)
    elif args.datamode == 'TCHW':
        enc: encoder.SparseEncoder = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False, in_chans=args.in_channels)
        dec = LightDecoder(enc.downsample_raito, sbn=args.sbn, out_chan=1)

    model_without_ddp = SparKFinetune(
        sparse_encoder=enc, dense_decoder=dec, mask_ratio=args.mask,
        densify_norm=args.densify_norm, sbn=args.sbn, datamode=args.datamode, transition=args.transition
    ).to(args.device)
    print(f'[FT model] model = {model_without_ddp}\n')
    
    # the model has been randomly initialized in their construction time
    # now try to load some checkpoint as model weight initialization; this ONLY loads the model weights
    misc.initialize_weight(args.init_weight, model_without_ddp)

    # if freeze parameters?
    if args.freeze == 'encoder':
        for p in model_without_ddp.sparse_encoder.parameters():
            p.requires_grad = False
        print('[FT] [freeze_model] encoder frozen')
    elif args.freeze == 'decoder':
        for p in model_without_ddp.dense_decoder.parameters():
            p.requires_grad = False
        print('[FT] [freeze_model] decoder frozen')
    elif args.freeze == 'all':
        for p in model_without_ddp.sparse_encoder.parameters():
            p.requires_grad = False
        for p in model_without_ddp.dense_decoder.parameters():
            p.requires_grad = False
        print('[FT] [freeze_model] encoder and decoder frozen')
    else:  # nothing to freeze
        print('[FT] [freeze_model] nothing to freeze')
    
    if dist.initialized():
        model: DistributedDataParallel = DistributedDataParallel(model_without_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    else:
        model = LocalDDP(model_without_ddp)
    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model_without_ddp, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
    opt_clz = {
        'sgd': partial(torch.optim.SGD, momentum=0.9, nesterov=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, args.ada)),
        'lamb': partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada), max_grad_norm=5.0),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=args.weight_decay)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume the experiment from some checkpoint.pth; this will load model weights, optimizer states, and last epoch (ep_start)
    # if loaded, ep_start will be greater than 0
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model_without_ddp, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [FT already done]    Min/Last Recon Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[FT start] from ep{ep_start}')
        
        ft_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = finetune_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer)
            Ltrain = stats['train_loss']
            Lmse = stats['mse_loss']
            Lssim = stats['ssim_loss']

            run['train/loss'].append(Ltrain)
            run['train/mse_loss'].append(Lmse)
            run['train/ssim_loss'].append(Lssim)

            # Evaluate on validation set
            conf_thresholds = [0.2, 0.5, 1.0]
            Lval, Lval_denorm_pixel, val_loss_single_pos, metric_all, metric_single = evaluate(args, args.device, itrt_val, iters_val, model, thresholds=conf_thresholds)

            run["eval/losspix/all"].append(Lval_denorm_pixel)
            for th in conf_thresholds:
                run[f"eval/csi/{th}/all"].append(metric_all[th]['csi'])
                run[f"eval/acc/{th}/all"].append(metric_all[th]['accuracy'])
                run[f"eval/f1/{th}/all"].append(metric_all[th]['f1'])
                run[f"eval/prec/{th}/all"].append(metric_all[th]['precision'])
                run[f"eval/recall/{th}/all"].append(metric_all[th]['recall'])
                run[f"eval/far/{th}/all"].append(metric_all[th]['far'])
                run[f"eval/hss/{th}/all"].append(metric_all[th]['hss'])
            for k in range(12):  # hard encoding
                run[f"eval/losspix/single/{k+1}"].append(val_loss_single_pos[k])
                for th in conf_thresholds:
                    run[f"eval/csi/{th}/single/{k+1}"].append(metric_single[th][k]['csi'])
                    run[f"eval/acc/{th}/single/{k+1}"].append(metric_single[th][k]['accuracy'])
                    run[f"eval/f1/{th}/single/{k+1}"].append(metric_single[th][k]['f1'])
                    run[f"eval/prec/{th}/single/{k+1}"].append(metric_single[th][k]['precision'])
                    run[f"eval/recall/{th}/single/{k+1}"].append(metric_single[th][k]['recall'])
                    run[f"eval/far/{th}/single/{k+1}"].append(metric_single[th][k]['far'])
                    run[f"eval/hss/{th}/single/{k+1}"].append(metric_single[th][k]['hss'])

            last_loss = Lval
            if last_loss < min_loss:
                misc.save_checkpoint_with_meta_info_and_opt_state(f'{args.model}_withdecoder_1kfinetuned_spark_style.pth', args, ep, performance_desc,model_without_ddp.state_dict(with_config=True), optimizer.state_dict())
                misc.save_checkpoint_model_weights_only(f'{args.model}_1kfinetuned_timm_style.pth', args, model_without_ddp.sparse_encoder.sp_cnn.state_dict())
                misc.save_checkpoint_model_weights_only(f'{args.model}_1kfinetuned_enc+dec_onlyweights_style.pth', args, model_without_ddp.state_dict())
            min_loss = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.8f} {last_loss:.8f}'

            ep_cost = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Loss Train: {Ltrain},   Loss Val(Pixel): {Lval_denorm_pixel},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.Ltrain = Ltrain
            args.Lval_pixel = Lval_denorm_pixel
            args.Lval_single_pixel = val_loss_single_pos
            # args.CSIval = metric_all["csi"]
            log_epoch(args)
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [FT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - ft_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        tb_lg.close()
        # time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    log_epoch(args)

    run.stop()


def finetune_one_ep(ep, args, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[FT] Epoch {ep}:'
    
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, (inp, tar) in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        cur_mask_ratio = mask_ratio_scheduler.mask_ratio_schedule(ep, args.ep, args.mask, args.mask_ratio_scheduler_type)

        # forward and backward
        inp = inp.to(args.device, non_blocking=True)
        tar = tar.to(args.device, non_blocking=True)
        # SparK.forward
        # TODO try l2 loss as optimization target, was 'recon_img' for original version
        ################################################################################################
        # _, _, recovered_bchw = model(inp, active_b1ff=None, return_mode='recon_img', datamode=args.datamode)
        # if args.datamode == 'TCHW':
        #     B, T, C, H, W = inp.shape
        #     recovered_btchw = recovered_bchw.view(B, T, C, H, W)
        #     pred = recovered_btchw.view(B, T*C, H, W)
        #     loss = mse_loss(pred, tar.view(B, T*C, H, W))  # L2 loss
        # elif args.datamode == 'CHW':
        #     loss = mse_loss(recovered_bchw, tar)
        ################################################################################################
        # loss = model(inp, active_b1ff=None, return_mode='l2_loss', datamode=args.datamode)
        ################################################################################################
        _, _, recovered_bchw = model(inp, active_b1ff=None, return_mode='recon_img', datamode=args.datamode)
        ssim = SSIM(data_range=(0., 1.)).to(args.device)
        if args.datamode == 'TCHW':
            B, T, C, H, W = inp.shape
            recovered_btchw = recovered_bchw.view(B, T, C, H, W)
            pred = recovered_btchw.view(B, T*C, H, W)
            mse_loss_val = mse_loss(pred, tar.view(B, T*C, H, W))  # L2 loss
            ssim_loss_val = 1 - ssim(pred, tar.view(B, T*C, H, W))
            loss = mse_loss_val + 1e-4 * ssim_loss_val
        elif args.datamode == 'CHW':
            mse_val = mse_loss(recovered_bchw, tar)
            ssim_loss_val = 1 - ssim(recovered_bchw, tar)
            loss = mse_val + 1e-4 * ssim_loss_val
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()
        
        # log
        me.update(mse_loss=mse_val.item())
        me.update(ssim_loss=ssim_loss_val.item())
        me.update(train_loss=loss)
        me.update(max_lr=max_lr)
        # tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        # tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        # tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        # tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        # tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


@torch.no_grad()
def evaluate(args, dev, itrt_val, iters_val, model, thresholds=None, monitor_pos=5):
    """
    Returns:
        val_loss
        val_loss_denorm_pixel
        val_csi
    """
    me = misc.MetricLogger(delimiter='  ')
    header = f'[FT] Eval :'
    training = model.training
    model.train(False)
    conf_matrices = {}
    for threshold in thresholds:
        if not threshold in conf_matrices.keys():
            conf_matrices[threshold] = {}
            conf_matrices[threshold]['all'] = {}
            for i in range(12):
                conf_matrices[threshold][i] = {}

        conf_matrices[threshold]['all']['tp'] = 0
        conf_matrices[threshold]['all']['fp'] = 0
        conf_matrices[threshold]['all']['tn'] = 0
        conf_matrices[threshold]['all']['fn'] = 0

        for i in range(12):  # hard encoding
            conf_matrices[threshold][i]['tp'] = 0
            conf_matrices[threshold][i]['fp'] = 0
            conf_matrices[threshold][i]['tn'] = 0
            conf_matrices[threshold][i]['fn'] = 0

    mse_error = 0.
    mse_error_at_monitor_pos = torch.zeros(12).to(dev)  # hard encoding
    for it, (inp, tar) in enumerate(me.log_every(iters_val, itrt_val, 3, header)):
        # all_target.extend(tar)
        inp = inp.to(dev, non_blocking=True)
        tar = tar.to(dev, non_blocking=True)
        _, _, oup = model(inp, active_b1ff=None, return_mode='recon_img', datamode=args.datamode)

        if args.datamode == 'TCHW':
            B, T, C, H, W = tar.shape
            tar = tar.view(B, T*C, H, W)
            oup = oup.view(B, T*C, H, W)

        it_val_loss = mse_loss(oup*47.18, tar*47.18, reduction='sum').item()
        it_val_loss_denorm_pixel = (it_val_loss / torch.numel(tar))
        it_val_loss_single_denorm = torch.sum((oup*47.18 - tar*47.18)**2, dim=(0, 2, 3))
        # logging
        me.update(val_loss=it_val_loss)
        me.update(val_loss_denorm_pixel=it_val_loss_denorm_pixel)

        # TODO compute mse, confusion matrix
        mse_error += it_val_loss
        mse_error_at_monitor_pos += it_val_loss_single_denorm
        for threshold in thresholds:
            oup, tar = oup * 12., tar * 12.
            pred_mask, tar_mask = oup > threshold, tar > threshold
            # cauculate confusion matrix with pred_mask and tar_mask
            ConfMat = ConfusionMatrix(task="binary", num_classes=2, threshold=threshold).to(dev)
            confmat = ConfMat(pred_mask, tar_mask)
            tn = confmat[0, 0].item()
            fp = confmat[0, 1].item()
            fn = confmat[1, 0].item()
            tp = confmat[1, 1].item()

            # compute confusion matrix for each channel (12 in total)
            for i in range(12):  # hard encoding
                confmat_single = ConfMat(pred_mask[:, i], tar_mask[:, i])
                tn_single = confmat_single[0, 0].item()
                fp_single = confmat_single[0, 1].item()
                fn_single = confmat_single[1, 0].item()
                tp_single = confmat_single[1, 1].item()

                conf_matrices[threshold][i]['tp'] += tp_single
                conf_matrices[threshold][i]['fp'] += fp_single
                conf_matrices[threshold][i]['tn'] += tn_single
                conf_matrices[threshold][i]['fn'] += fn_single

            # csi_threshold.update(oup, tar)
            conf_matrices[threshold]['all']['tp'] += tp
            conf_matrices[threshold]['all']['fp'] += fp
            conf_matrices[threshold]['all']['tn'] += tn
            conf_matrices[threshold]['all']['fn'] += fn

    model.train(training)
    # print("stacking all tensors")
    # all_target_tensor, all_pred_tensor = torch.stack(all_target).to(dev), torch.stack(all_pred).to(dev)
    # t = torch.concat([all_target_tensor, all_pred_tensor]).to(dev)
    # tdist.all_reduce(t)

    # all_target_tensor, all_pred_tensor = t[0], t[1]
    print('calculate val_loss')
    val_loss = mse_error / 1557
    print('calculate val_loss_denorm_pixel')
    val_loss_denorm_pixel = mse_error / (1557*12*288*288)
    val_loss_pix_single = mse_error_at_monitor_pos / (1557*288*288)
    val_loss_pix_single = val_loss_pix_single.cpu().tolist()
    print('calculate confusion stat')
    # calculate csi depending on the conf_matrices

    confusion_stat_all, confusion_stat_single = {}, {}

    for threshold in thresholds:
        tp = conf_matrices[threshold]['all']['tp']
        fp = conf_matrices[threshold]['all']['fp']
        tn = conf_matrices[threshold]['all']['tn']
        fn = conf_matrices[threshold]['all']['fn']

        if not threshold in confusion_stat_all.keys():
            confusion_stat_all[threshold] = {}
        if not threshold in confusion_stat_single.keys():
            confusion_stat_single[threshold] = {}
            for i in range(12):
                confusion_stat_single[threshold][i] = {}

        confusion_stat_all[threshold]['accuracy'] = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        confusion_stat_all[threshold]['csi'] = tp / (tp + fn + fp + 1e-8)
        confusion_stat_all[threshold]['far'] = fp / (tp + fp + 1e-8)
        confusion_stat_all[threshold]['f1'] = 2 * tp / (2 * tp + fp + fn + 1e-8)
        confusion_stat_all[threshold]['precision'] = tp / (tp + fp + 1e-8)
        confusion_stat_all[threshold]['recall'] = tp / (tp + fn + 1e-8)
        confusion_stat_all[threshold]['hss'] = (tp*tn - fn*fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-8)

        for i in range(12):  # hard encoding
            tp_single = conf_matrices[threshold][i]['tp']
            fp_single = conf_matrices[threshold][i]['fp']
            tn_single = conf_matrices[threshold][i]['tn']
            fn_single = conf_matrices[threshold][i]['fn']

            confusion_stat_single[threshold][i]['accuracy'] = (tp_single + tn_single) / (tp_single + tn_single + fp_single + fn_single + 1e-8)
            confusion_stat_single[threshold][i]['csi'] = tp_single / (tp_single + fn_single + fp_single + 1e-8)
            confusion_stat_single[threshold][i]['far'] = fp_single / (tp_single + fp_single + 1e-8)
            confusion_stat_single[threshold][i]['f1'] = 2 * tp_single / (2 * tp_single + fp_single + fn_single + 1e-8)
            confusion_stat_single[threshold][i]['precision'] = tp_single / (tp_single + fp_single + 1e-8)
            confusion_stat_single[threshold][i]['recall'] = tp_single / (tp_single + fn_single + 1e-8)
            confusion_stat_single[threshold][i]['hss'] = (tp_single*tn_single - fn_single*fp_single) / ((tp_single + fn_single) * (fn_single + tn_single) + (tp_single + fp_single) * (fp_single + tn_single) + 1e-8)

    me.synchronize_between_processes()
    return val_loss, val_loss_denorm_pixel, val_loss_pix_single, confusion_stat_all, confusion_stat_single



if __name__ == '__main__':
    main_finetune(args=arg_util.get_args_parser().parse_args())
