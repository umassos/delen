#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    """ 1x1 convolutional layer """
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int=1, padding=1) -> nn.Conv2d:
    """ 3x3 convolutional layer """
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=padding, bias=False)


def get_same_padding_conv2d(in_channels: int,
                            out_channels: int,
                            kernel_size:int = 1,
                            groups: int = 1,
                            stride: int = 1,
                            p: int = None,
                            bias: bool = False) -> nn.Conv2d:
    """ Same padding Conv2d """
    if not p:
        p = (kernel_size-1) // 2
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, groups=groups, bias=bias, padding=p)


def drop_connect(inputs: torch.Tensor, p: float, training: bool):
    """Drop connect """
    if not training: return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output
