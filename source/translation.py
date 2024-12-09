import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Block

from models.openstl.models.simvp_model import MidMetaNet
from models.openstl.models.convlstm_model import ConvLSTM_Model
from models.openstl.models.convlstm_simple import ConvLSTM_ModelSimple


class TranslationBase(nn.Module):
    """
    Base class for transition models
    A transition network receives a list of features of past and returns a list of feature of future
    input: {[B, T, C0, f0, f0], [B, T, C1, f1, f1], ...} (past features)
    output: {[B, T, C0, f0, f0], [B, T, C1, f1, f1], ...} (future features)
    """

    def __init__(self):
        super(TranslationBase, self).__init__()

    def forward(self, x, fea_list, mode, **kwargs):
        """
        x [B, T, C, H, W]: input images
        fea_list: {[B, T, C0, f0, f0], ... [B, T, Cn, fn, fn]} (n-level features, here n=4)
        return: {[B, T, C0, f0, f0], ... [B, T, Cn, fn, fn]} (n-level features, here n=4)
        """
        assert hasattr(self, 'net'), 'Transition network not defined or is not named as "net"'
        for i, fea in enumerate(fea_list):
            if fea is not None:
                if mode == 'CHW':
                    fea_list[i] = self.net[i](fea)
                elif mode == 'TCHW':
                    assert 'btchw' in kwargs, 'shape BTCHW not in kwargs'
                    B, T, _, _, _ = kwargs['btchw']
                    _, C, F, _ = fea.shape
                    fea_list[i] = (self.net[i](fea.view(B, T, fea.shape[-3], fea.shape[-2], fea.shape[-1]))).view(B*T, C, F, F)
                else:
                    raise ValueError(f"mode {mode} not supported")
        return fea_list


class TranslationIdentity(TranslationBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, fea_list, mode, **kwargs):
        return fea_list


class ResidualBlock(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return x + self.network(x)


# Linear Conv+tanh
class TranslationLinearA(TranslationBase):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )
        ])
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.trunc_normal_(m.bias, std=0.02)


def build_translation_network(transition_name, **kwargs):
    print(f"Building translation network {transition_name}")
    if transition_name == 'linearA':
        return TranslationLinearA()
    elif transition_name == 'identity':
        return TranslationIdentity()
    else:
        raise NotImplementedError(f"Translation network {transition_name} not implemented")
