__version__ = "0.6.1"
from .model import EfficientNet,EfficientNet_v2,EfficientNet_v3
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)

def efficientnet_b0(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b0",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b0", kwargs)

def efficientnet_b1(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b1",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b1", kwargs)

def efficientnet_b2(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b2",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b2", kwargs)

def efficientnet_b3(pretrained=False, **kwargs):
    if pretrained:
        return  EfficientNet.from_pretrained("efficientnet-b3",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b3", kwargs)

def efficientnet_b4(pretrained=False, **kwargs):
    if pretrained:
        if kwargs.get('use_feature'):
             kwargs.pop('use_feature')
             return  EfficientNet_v2.from_pretrained("efficientnet-b4",
                advprop=True, **kwargs)
        if kwargs.get('only_use_feature'):
             kwargs.pop('only_use_feature')
             print('only_use_feature')
             return  EfficientNet_v3.from_pretrained("efficientnet-b4",
                advprop=True, **kwargs)
        return  EfficientNet.from_pretrained("efficientnet-b4",
                advprop=True, **kwargs)
    else:
        return EfficientNet.from_name("efficientnet-b4", kwargs)




