from pprint import pformat
from typing import List, Optional

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder
from source.translation import build_translation_network


class SparKFinetune(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='bn', sbn=False, mask_strategy='random', datamode=None, transition=None
    ):
        super().__init__()
        input_size, downsample_raito = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito = downsample_raito
        self.fmap_h, self.fmap_w = input_size // downsample_raito, input_size // downsample_raito
        self.mask_strategy = mask_strategy
        self.mask_ratio = mask_ratio
        self.datamode = datamode

        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        # TODO fill it with linear projection first

        self.transition = build_translation_network(transition)

        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()

        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(
                self.hierarchy):  # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)

            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'ln':
                densify_norm = encoder.SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)

            # create densify proj
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()  # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         bias=True)
                print(
                    f'[SparK.__init__, densify {i + 1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)

            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2

        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')

        # these are deprecated and would never be used; can be removed.
        self.register_buffer('imn_m', torch.empty(1, 3, 1, 1))
        self.register_buffer('imn_s', torch.empty(1, 3, 1, 1))
        self.register_buffer('norm_black', torch.zeros(1, 3, input_size, input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...

    def mask(self, inp_bchw, device, generator=None, mask_ratio: Optional[float] = None):
        B, C = inp_bchw.shape[0], inp_bchw.shape[1]
        h, w = self.fmap_h, self.fmap_w
        if self.mask_strategy == 'random':
            idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
            mask_ratio = self.mask_ratio if mask_ratio is None else mask_ratio
            len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
            idx = idx[:, :len_keep].to(device)  # (B, len_keep)
            return (torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w))
        elif self.mask_strategy == 'threshold':
            thresholds = torch.arange(start=0., end=2., step=0.1, device=device) / 47.18
            # create mask depending on the threshold
            mask = inp_bchw > thresholds.repeat(B, C, h, w, 1).view(len(thresholds), B, C, h, w)
            return mask

    def forward(self, inp_bchw_or_btchw: torch.Tensor, active_b1ff=None, vis=False, return_mode='recon_img', datamode='CHW'):
        # step0. Prepare the input shape
        # inp_btchw_backup = inp_btchw.clone()  # (B, T, C, H, W)
        # from here on, inp_bchw means inp_b(t*c)hw
        if datamode == 'TCHW':
            assert inp_bchw_or_btchw.dim() == 5, f'input shape should be (B, T, C, H, W), but got {inp_bchw_or_btchw.shape}'
            B, T, C, H, W = inp_bchw_or_btchw.shape
            # inp_bchw_or_btchw = inp_bchw_or_btchw.view(B * T, C, H, W)
        elif datamode == 'CHW':
            assert inp_bchw_or_btchw.dim() == 4, f'input shape should be (B, C, H, W), but got {inp_bchw_or_btchw.shape}'
            # inp_bchw = inp_bchw_or_btchw
        else:
            raise ValueError(f'invalid datamode={datamode}, should be one of ("TCHW", "CHW")')

        # step1. Mask
        if active_b1ff is None:  # rand mask
            # TODO: use same mask for all images in the same sequence, now it's different for each
            active_b1ff: torch.BoolTensor = \
                self.mask(inp_bchw_or_btchw.view(B*T, C, H, W) if datamode == 'TCHW' else inp_bchw_or_btchw, inp_bchw_or_btchw.device)  # (B, 1, f, f)
        encoder._cur_active = active_b1ff  # (B, 1, f, f)
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_raito, 2).repeat_interleave(self.downsample_raito, 3)  # (B, 1, H, W)
        masked_bchw_or_btchw = inp_bchw_or_btchw.view(B*T, C, H, W) * active_b1hw if datamode == 'TCHW' else inp_bchw_or_btchw * active_b1hw

        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        fea_bcff_or_btcff_list: List[torch.Tensor] = self.sparse_encoder(masked_bchw_or_btchw.view(B*T, C, H, W) if datamode == 'TCHW' else masked_bchw_or_btchw)
        fea_bcff_or_btcff_list.reverse()  # after reversion: from the smallest feature map to the largest

        btchw_info = None if datamode == "CHW" else [B, T, C, H, W]
        fea_bcff_or_btcff_list = self.transition(inp_bchw_or_btchw, fea_bcff_or_btcff_list, mode=datamode, btchw=btchw_info)

        # step3. Densify: get hierarchical dense features for decoding
        cur_active = active_b1ff  # (B, 1, f, f)
        to_dec = []
        for i, bcff in enumerate(fea_bcff_or_btcff_list):  # from the smallest feature map to the largest
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)  # fill in empty (non-active) positions with [mask] tokens
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)  # dilate the mask map, from (B, 1, f, f) to (B, 1, H, W)

        # step4. Decode and reconstruct
        rec_bchw = self.dense_decoder(to_dec)

        # step 4.1, reshape the output
        # rec_btchw = rec_bchw.view(B, T, C, H, W)

        inp, rec = \
            (self.patchify(inp_bchw_or_btchw.view(B * T, C, H, W) if datamode == 'TCHW' else inp_bchw_or_btchw),
             self.patchify(rec_bchw))  # inp and rec: (B, L = f*f, N = C*downsample_raito**2)
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)  # (B, L, C) ==mean==> (B, L)

        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  # (B, 1, f, f) => (B, L)
        # masked_bchw = inp_bchw_or_btchw * active_b1hw

        if return_mode == 'l2_loss':
            return l2_loss.sum() / l2_loss.shape[0]
        elif return_mode == 'recon_loss':
            recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8)  # loss only on masked (non-active) patches
            return recon_loss
        elif return_mode == 'recon_img':
            rec_bchw = self.unpatchify(rec * var + mean)
            return inp_bchw_or_btchw, masked_bchw_or_btchw, rec_bchw
        elif return_mode == 'recon_or_inp_img':
            rec_bchw = self.unpatchify(rec * var + mean)
            rec_or_inp = torch.where(active_b1hw, inp_bchw_or_btchw, rec_bchw)
            return inp_bchw_or_btchw, masked_bchw_or_btchw, rec_or_inp
        else:
            raise ValueError(f'invalid return_mode={return_mode}')

    def patchify(self, bchw):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  # (B, f*f, 3*downsample_raito**2)
        return bln

    def unpatchify(self, bln):
        p = self.downsample_raito
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw

    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparKFinetune, self).__repr__().replace(SparKFinetune.__name__, "")}'
        )

    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,

            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparKFinetune, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state

    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparKFinetune, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
