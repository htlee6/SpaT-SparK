from timm.models import ResNet
from timm.models.registry import register_model
from timm.models.resnet import Bottleneck


@register_model
def resnet50_chan1(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_chans=1,
        **kwargs
    )
    return model


@register_model
def resnet50_chan12(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        in_chans=12,
        **kwargs
    )
    return model


@register_model
def resnet18_chan1(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        in_chans=1,
        **kwargs
    )
    return model


@register_model
def resnet18_chan12(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[2, 2, 2, 2],
        in_chans=12,
        **kwargs
    )
    return model


@register_model
def resnet10_chan12(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[1, 1, 1, 1],
        in_chans=12,
        **kwargs
    )
    return model


@register_model
def resnet10_chan1(pretrained=False, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[1, 1, 1, 1],
        in_chans=1,
        **kwargs
    )
    return model
