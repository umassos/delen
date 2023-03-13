#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, conv3x3, get_same_padding_conv2d, drop_connect
from typing import List, Tuple, Dict, Callable, Any


BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

GLOBAL_PARAMS = GlobalParams(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        image_size=32,
        dropout_rate=.3,
        num_classes=1000,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        drop_connect_rate=.3,
        depth_divisor=8,
        min_depth=None,
        include_top=True,
    )


EFFICIENTNET_B0_BLOCK_ARGS = [
    BlockArgs(num_repeat=1, kernel_size=3, stride=1, expand_ratio=1, input_filters=32,
              output_filters=16, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=2, kernel_size=3, stride=2, expand_ratio=6, input_filters=16,
              output_filters=24, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=2, kernel_size=5, stride=2, expand_ratio=6, input_filters=24,
              output_filters=40, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=3, kernel_size=3, stride=2, expand_ratio=6, input_filters=40,
              output_filters=80, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=3, kernel_size=5, stride=1, expand_ratio=6, input_filters=80,
              output_filters=112, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=4, kernel_size=5, stride=2, expand_ratio=6, input_filters=112,
              output_filters=192, se_ratio=.25, id_skip=True),
    BlockArgs(num_repeat=1, kernel_size=3, stride=1, expand_ratio=6, input_filters=192,
              output_filters=320, se_ratio=.25, id_skip=True)
]


def attention(attention_digits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ Give attention to channels """
    return torch.sigmoid(attention_digits) * x


def id_skip(identity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ Identity skip """
    return identity + x


class MBConvBlock(nn.Module):
    """ Mobile Inverted Residual Bottleneck Block """

    def __init__(self,
                 block_args: BlockArgs,
                 global_params: GlobalParams) -> None:
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self._id_skip = block_args.id_skip

        # Expansion
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        self._input_channel = inp
        self._expan_channel = oup
        self.layer_names = []
        self.layer_profile = {}

        if self._block_args.expand_ratio != 1:
            self._expand_conv = conv1x1(inp, oup)

            self.layer_names.append("_expand_conv")
            self.layer_profile["_expand_conv"] = {
                "exe_time"       : [],
                "input_channels" : inp,
                "output_channels": oup
            }

            self._bn0 = nn.BatchNorm2d(self._expan_channel, momentum=self._bn_mom, eps=self._bn_eps)
            self.layer_names.append("_bn0")
            self.layer_profile["_bn0"] = {
                "exe_time"       : [],
                "input_channels" : oup,
                "output_channels": oup
            }

            self.layer_names.append("_silu0")
            self.layer_profile["_silu0"] = {
                "exe_time"       : [],
                "input_channels" : oup,
                "output_channels": oup
            }

        # Depthwise convolution
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = get_same_padding_conv2d(in_channels=oup, out_channels=oup,
                                                       groups=oup, kernel_size=k, stride=s)

        self.layer_names.append("_depthwise_conv")
        self.layer_profile["_depthwise_conv"] = {
            "exe_time"       : [],
            "input_channels" : oup,
            "output_channels": oup
        }

        self._bn1 = nn.BatchNorm2d(self._expan_channel, momentum=self._bn_mom, eps=self._bn_eps)
        self.layer_names.append("_bn1")
        self.layer_profile["_bn1"] = {
            "exe_time"       : [],
            "input_channels" : oup,
            "output_channels": oup
        }

        self.layer_names.append("_silu1")
        self.layer_profile["_silu1"] = {
            "exe_time"       : [],
            "input_channels" : oup,
            "output_channels": oup
        }

        # Squeeze and Excitation
        if self.has_se:
            self.layer_names.append("_avg_pool_se")
            self.layer_profile["_avg_pool_se"] = {
                "exe_time"       : [],
                "input_channels" : oup,
                "output_channels": oup
            }
            num_squeezed_channels = int(self._block_args.input_filters * self._block_args.se_ratio)
            self._squeeze_channels = num_squeezed_channels

            squeeze_channel = max(1, int(self._squeeze_channels))
            se_reduce = get_same_padding_conv2d(in_channels=self._expan_channel,
                                                out_channels=squeeze_channel,
                                                kernel_size=1)
            self.layer_names.append("_se_reduce")
            self.layer_profile["_se_reduce"] = {
                "exe_time"       : [],
                "input_channels" : self._expan_channel,
                "output_channels": squeeze_channel
            }
            self.layer_names.append("_silu_se")
            self.layer_profile["_silu_se"] = {
                "exe_time"       : [],
                "input_channels" : oup,
                "output_channels": oup
            }

            se_expand = get_same_padding_conv2d(in_channels=squeeze_channel,
                                                out_channels=self._expan_channel,
                                                kernel_size=1)
            self.layer_names.append("_se_expand")
            self.layer_profile["_se_expand"] = {
                "exe_time"       : [],
                "input_channels" : squeeze_channel,
                "output_channels": self._expan_channel
            }

            self.layer_names.append("_se_scale")
            self.layer_profile["_se_scale"] = {
                "exe_time"       : [],
                "input_channels" : self._expan_channel,
                "output_channels": self._expan_channel
            }

            self._se_reduce = se_reduce
            self._se_expand = se_expand

        # Pointwise convolution
        final_oup = self._block_args.output_filters
        self._final_channel = final_oup
        self._project_conv = get_same_padding_conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1)
        self.layer_names.append("_project_conv")
        self.layer_profile["_project_conv"] = {
            "exe_time"       : [],
            "input_channels" : oup,
            "output_channels": final_oup
        }

        self._bn2 = nn.BatchNorm2d(self._final_channel, momentum=self._bn_mom, eps=self._bn_eps)
        self.layer_names.append("_bn2")
        self.layer_profile["_bn2"] = {
            "exe_time"       : [],
            "input_channels" : final_oup,
            "output_channels": final_oup
        }

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self._id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            self.layer_names.append("_id_skip")
            self.layer_profile["_id_skip"] = {
                "exe_time"       : [],
                "input_channels" : input_filters,
                "output_channels": input_filters
            }

        self._silu = nn.SiLU(inplace=True)

    def _wrapper(self, name: str, fn: Callable, *args) -> Any:
        """
        Perform pre-process and post-process
        Args:
            name:  name of this function
            fn:  Target function
            args:  Function arguments

        Returns:
            ret: return from the target function

        """
        start_t = time.time()
        ret = fn(*args)
        self.layer_profile[name]["exe_time"].append(time.time() - start_t)

        return ret

    def forward(self, inputs: torch.Tensor, drop_connect_rate=None):
        """
        MBConv forward function
        Args:
            inputs: input_tensor
            drop_connect_rate: drop depth

        Returns:
            out: (output_tensor, width) tuple
        """
        x = inputs

        if self._block_args.expand_ratio != 1:
            # Expand layer
            x = self._wrapper("_expand_conv", self._expand_conv, inputs)

            # Batch norm 0 layer
            x = self._wrapper("_bn0", self._bn0, x)

            # Swish 0 layer
            x = self._wrapper("_silu0", self._silu, x)

        # Depthwise convolutional layer
        x = self._wrapper("_depthwise_conv", self._depthwise_conv, x)

        # Batch nrom 1 layer
        x = self._wrapper("_bn1", self._bn1, x)

        # Swish 1 layer
        x = self._wrapper("_silu1", self._silu, x)

        # Squeeze and Excitation
        if self.has_se:
            # SE avg pool layer
            x_squeeze = self._wrapper("_avg_pool_se", F.adaptive_avg_pool2d, x, 1)

            # SE reduce layer
            x_squeeze = self._wrapper("_se_reduce", self._se_reduce, x_squeeze)

            # SE swish layer
            x_squeeze = self._wrapper("_silu_se", self._silu, x_squeeze)

            # SE expand layer
            x_squeeze = self._wrapper("_se_expand", self._se_expand, x_squeeze)

            # SE scale layer
            x = self._wrapper("_se_scale", attention, x_squeeze, x)

        # Pointwise Convolution
        x = self._wrapper("_project_conv", self._project_conv, x)

        # Batch norm 2 layer
        x = self._wrapper("_bn2", self._bn2, x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self._id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            # Skip connection layer
            x = self._wrapper("_id_skip", id_skip, inputs, x)

        return x


class EfficientNet(nn.Module):
    """ Anytime EfficientNet-B0 model """
    def __init__(self, block_args: List[BlockArgs] = EFFICIENTNET_B0_BLOCK_ARGS,
                 global_params: GlobalParams = GLOBAL_PARAMS):
        super(EfficientNet, self).__init__()
        self._global_params = global_params
        self._block_args = block_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        self.layer_names = []
        self.layer_profile = {}

        # Stem
        in_channels = 3
        out_channels = 32
        self._stem_channel = out_channels
        self._conv_stem = get_same_padding_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                                  stride=2)
        self.layer_names.append("_conv_stem")
        self.layer_profile["_conv_stem"] = {
            "exe_time"       : [],
            "input_channels" : in_channels,
            "output_channels": out_channels
        }

        self._bn0 = nn.BatchNorm2d(self._stem_channel, momentum=bn_mom, eps=bn_eps)
        self.layer_names.append("_bn0")
        self.layer_profile["_bn0"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": out_channels
        }

        self.layer_names.append("_silu0")
        self.layer_profile["_silu0"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": out_channels
        }

        # Build blocks
        self._blocks = nn.ModuleList([])

        for i, block_args in enumerate(self._block_args):
            self._blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            for _ in range(block_args.num_repeat-1):
                self._blocks.append(MBConvBlock(block_args, global_params))

        # Head
        in_channels = block_args.output_filters
        out_channels = 1280
        self._head_in_channel = in_channels
        self._head_channel = out_channels
        self._conv_head = get_same_padding_conv2d(in_channels, out_channels, kernel_size=1)
        self.layer_names.append("_conv_head")
        self.layer_profile["_conv_head"] = {
            "exe_time"       : [],
            "input_channels" : in_channels,
            "output_channels": out_channels
        }

        self._bn1 = nn.BatchNorm2d(self._head_channel, momentum=bn_mom, eps=bn_eps)
        self.layer_names.append("_bn1")
        self.layer_profile["_bn1"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": out_channels
        }

        self.layer_names.append("_silu1")
        self.layer_profile["_silu1"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": out_channels
        }

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.layer_names.append("_avg_pooling")
        self.layer_profile["_avg_pooling"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": out_channels
        }

        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(self._head_channel, self._global_params.num_classes)
        self.layer_names.append("_fc")
        self.layer_profile["_fc"] = {
            "exe_time"       : [],
            "input_channels" : out_channels,
            "output_channels": self._global_params.num_classes
        }

        self._silu = nn.SiLU(inplace=True)

    def _wrapper(self, name: str, fn: Callable, *args) -> Any:
        """
        Perform pre-process and post-process
        Args:
            name:  name of this function
            fn:  Target function
            args:  Function arguments

        Returns:
            ret: return from the target function

        """
        start_t = time.time()
        ret = fn(*args)
        self.layer_profile[name]["exe_time"].append(time.time() - start_t)

        return ret

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward function
        Args:
            inputs: input tensor

        Returns:

        """

        # Stem
        x = self._wrapper("_conv_stem", self._conv_stem, inputs)

        # Batch norm 0
        x = self._wrapper("_bn0", self._bn0, x)

        # Swish 0
        x = self._wrapper("_silu0", self._silu, x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)

            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._wrapper("_conv_head", self._conv_head, x)

        # Batch norm 1
        x = self._wrapper("_bn1", self._bn1, x)

        # Swish layer 1
        x = self._wrapper("_silu1", self._silu, x)

        # Pooling
        x = self._wrapper("_avg_pooling", self._avg_pooling, x)

        # Fully connected layer
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._wrapper("_fc", self._fc, x)

        return x

    def latency_summary(self):
        """ Summary the execution time of each layer """
        if not self.layer_profile["_conv_stem"]["exe_time"]:
            print("Run at least one inference to get the summary")
            return

        head_format = "{:30s} {:>15s} {:>15s} {:>10s}"
        total = 0
        print(head_format.format("Layers", "In-channels", "Out-channels", "Time (ms)"))
        print("-"*90)
        total += self._print_layer_profile("_conv_stem", self.layer_profile)
        total += self._print_layer_profile("_bn0", self.layer_profile)
        total += self._print_layer_profile("_silu0", self.layer_profile)

        for idx, block in enumerate(self._blocks):
            for layer_name in block.layer_names:
                total += self._print_layer_profile(layer_name, block.layer_profile, prefix="block_%d" % idx)

        total += self._print_layer_profile("_conv_head", self.layer_profile)
        total += self._print_layer_profile("_bn1", self.layer_profile)
        total += self._print_layer_profile("_silu1", self.layer_profile)
        total += self._print_layer_profile("_avg_pooling", self.layer_profile)
        total += self._print_layer_profile("_fc", self.layer_profile)

        print("-"*90)
        print(head_format.format("total", "", "", "%.2f" % total, ""))

    def _print_layer_profile(self, layer_name, layer_profile, prefix=""):
        """ print layer profile as a row """
        row_format = "{:30s} {:15d} {:15d} {:10.2f}"
        exe_time = np.mean(layer_profile[layer_name]["exe_time"]) * 1000
        print(row_format.format(prefix + layer_name,
                                layer_profile[layer_name]["input_channels"],
                                layer_profile[layer_name]["output_channels"],
                                exe_time))
        return exe_time

    def _export_layer_profile(self, layer_name, layer_profile, prefix="", **kwargs):
        """ print layer profile as a list """
        records = []
        for i in range(len(layer_profile[layer_name]["exe_time"])):
            t = layer_profile[layer_name]["exe_time"][i]

            records.append({
                "layer_name"  : prefix + layer_name,
                "in_channels" : layer_profile[layer_name]["input_channels"],
                "out_channels": layer_profile[layer_name]["output_channels"],
                "exe_time"    : t * 1000,
                **kwargs
            })

        return records

    def export_profile(self, **kwargs) -> List:
        """ Export layer profile to a list of dictionary """
        profile = []
        profile.extend(self._export_layer_profile("_conv_stem", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_bn0", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_silu0", self.layer_profile, **kwargs))

        for idx, block in enumerate(self._blocks):
            for layer_name in block.layer_names:
                profile.extend(self._export_layer_profile(layer_name, block.layer_profile, prefix="block_%d" % idx,
                                                          **kwargs))

        profile.extend(self._export_layer_profile("_conv_head", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_bn1", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_silu1", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_avg_pooling", self.layer_profile, **kwargs))
        profile.extend(self._export_layer_profile("_fc", self.layer_profile, **kwargs))

        return profile


