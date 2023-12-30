import torch
from torch import Tensor
from typing import Optional
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


# [Private]
# A replacement for the Conv2d._conv_forward method that pads axes asymmetrically.
# This replacement method performs the same operation (as of torch v1.12.1+cu113), but it pads the X and Y axes separately based on the members
#   padding_modeX (string, either 'circular' or 'constant')
#   padding_modeY (string, either 'circular' or 'constant')
#   paddingX (tuple, cached copy of _reversed_padding_repeated_twice with the last two values zeroed)
#   paddingY (tuple, cached copy of _reversed_padding_repeated_twice with the first two values zeroed)
def __replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res


def get_layers_of_type(module, type_):
    layers = []
    all_layers = flatten(module)
    for layer in all_layers:
        if type(layer) == torch.nn.Conv2d:
            layers.append(layer)
    return layers


def set_layer_tiling(layer, tileX, tileY):
    assert isinstance(layer, torch.nn.Conv2d), "layer is not a Conv2d"
    layer.padding_modeX = "circular" if tileX else "constant"
    layer.padding_modeY = "circular" if tileY else "constant"
    layer.paddingX = (
        layer._reversed_padding_repeated_twice[0],
        layer._reversed_padding_repeated_twice[1],
        0,
        0,
    )
    layer.paddingY = (
        0,
        0,
        layer._reversed_padding_repeated_twice[2],
        layer._reversed_padding_repeated_twice[3],
    )
    layer._conv_forward = __replacementConv2DConvForward.__get__(layer, Conv2d)


def set_module_tiling(module, tileX, tileY, type_=torch.nn.Conv2d):
    layers = get_layers_of_type(module, type_)
    for layer in layers:
        set_layer_tiling(layer, tileX, tileY)
