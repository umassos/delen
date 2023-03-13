#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anytime_common import AnytimeSubNetwork
from .modules import conv1x1, conv3x3, get_same_padding_conv2d, drop_connect
from .measure import (
    _count_flops_conv2d,
    _count_flops_batchnorm2d,
    _count_flops_relu,
    _count_flops_linear,
    _count_flops_depthwise_conv2d
)
from torch.hub import load_state_dict_from_url
from typing import List, Tuple, Union, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

PARAM_DICT = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }

MODEL_URLS = {
    'efficientnet-b0':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'efficientnet-b2':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'efficientnet-b6':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7':
        'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}


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


def get_global_param(model_name):
    """ Return GlobalParam for a given model """
    width_coefficient, depth_coefficient, image_size, dropout_rate = PARAM_DICT[model_name]

    global_params = GlobalParams(
            width_coefficient=width_coefficient,
            depth_coefficient=depth_coefficient,
            image_size=224,
            dropout_rate=dropout_rate,
            num_classes=365,
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            drop_connect_rate=.3,
            depth_divisor=8,
            min_depth=None,
            include_top=True,
    )
    return global_params


def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters

    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Calculate module's repeat number of a block based on depth multiplier.
       Use depth_coefficient of global_params.
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.
    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = get_same_padding_conv2d(in_channels=inp, out_channels=oup, kernel_size=1)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = get_same_padding_conv2d( in_channels=oup, out_channels=oup,
                                                        groups=oup, kernel_size=k, stride=s)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = get_same_padding_conv2d(in_channels=oup, out_channels=num_squeezed_channels,
                                                      kernel_size=1, bias=True)
            self._se_expand = get_same_padding_conv2d(in_channels=num_squeezed_channels, out_channels=oup,
                                                      kernel_size=1, bias=True)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        self._project_conv = get_same_padding_conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = nn.SiLU(inplace=True)

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).
        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class AnytimeEfficientNetDepth(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.
    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._output_layers = nn.ModuleDict()
        self._output_layer_index = []
        self._sub_networks = []
        self._exit_blocks = [2, 4]

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = get_same_padding_conv2d(in_channels, out_channels, kernel_size=3, stride=2)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._swish = nn.SiLU(inplace=True)

        sub_network = [self._conv_stem, self._bn0, self._swish]
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_idx, block_args in enumerate(self._blocks_args):
            block_stride = block_args.stride

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            sub_network.append(self._blocks[-1])

            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
                sub_network.append(self._blocks[-1])

            # Put an exit for each resolution shrink
            if block_idx in self._exit_blocks:
                exit_index = str(len(self._blocks) - 1)
                self._output_layers[exit_index] = nn.Sequential(
                        self._avg_pooling,
                        nn.Flatten(start_dim=1),
                        self._dropout,
                        nn.Linear(block_args.output_filters, self._global_params.num_classes)
                )
                self._output_layer_index.append(exit_index)
                self._sub_networks.append(nn.Sequential(*sub_network))
                sub_network = []

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = get_same_padding_conv2d(in_channels, out_channels, kernel_size=1)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        sub_network.append(self._conv_head)
        sub_network.append(self._bn1)
        sub_network.append(self._swish)
        self._sub_networks.append(nn.Sequential(*sub_network))

        self._classifier = nn.Sequential(
                        self._avg_pooling,
                        nn.Flatten(start_dim=1),
                        self._dropout,
                        nn.Linear(out_channels, self._global_params.num_classes)
                )
        self._output_layers[str(len(self._blocks)+1)] = self._classifier
        self._output_layer_index.append(str(len(self._blocks)+1))

    @property
    def num_subnetworks(self):
        return len(self._output_layers)

    @property
    def num_blocks(self):
        return len(self._blocks)

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this model after processing.
        """
        early_outputs = []

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

            if str(idx) in self._output_layers:
                early_outputs.append(self._output_layers[str(idx)](x))

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        # Pooling and final linear layer
        x = self._classifier(x)

        early_outputs.append(x)
        return early_outputs

        # if self.training:
        #     early_outputs.append(x)
        #     return early_outputs
        # else:
        #     return x

    def get_subnetwork(self, idx: int) -> nn.Module:
        """ Return get subnetwork given index """
        if idx < 0 or idx >= self.num_subnetworks:
            raise ValueError("Sub-network index out of range, available sub-networks {:d}, got {:d}".format(
                    self.num_subnetworks, idx))

        include_features = idx < self.num_subnetworks - 1
        return AnytimeSubNetwork(self._sub_networks[idx],
                                 self._output_layers[self._output_layer_index[idx]],
                                 include_features)

    def get_block(self, idx: int) -> Union[nn.Module, nn.Sequential]:
        """
        Return specific block as a module
        Args:
            idx: block index

        Returns:
            block
        """
        if idx < 0 or idx >= self.num_blocks:
            raise ValueError("Block index out of range, available blocks {:d}, got {:d}".format(self.num_blocks, idx))

        if idx == 0:
            return nn.Sequential(
                    self._conv_stem,
                    self._bn0,
                    self._swish,
                    self._blocks[idx]
            )
        elif idx == self.num_blocks - 1:
            return nn.Sequential(
                    self._blocks[idx],
                    self._conv_head,
                    self._bn1,
                    self._swish,
                    self._classifier
            )
        else:
            return self._blocks[idx]

    def restore_regular_model(self) -> nn.Sequential:
        """
        Restore to a regular DNN model with only the last exit
        Returns:
            model: regular DNN with only the last output layer

        """
        return nn.Sequential(
                self._conv_stem,
                self._bn0,
                self._swish,
                *self._blocks,
                self._conv_head,
                self._bn1,
                self._swish,
                self._classifier
        )


class AnytimeMBConvBlock(nn.Module):
    """ Mobile Inverted Residual Bottleneck Block """

    def __init__(self,
                 block_args: BlockArgs,
                 global_params: GlobalParams,
                 width: int = 1) -> None:
        super(AnytimeMBConvBlock, self).__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self._id_skip = block_args.id_skip
        self._width = width

        # Expansion
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        self._input_channel_per_subnet = inp // width
        self._expan_channel_per_subnet = oup // width

        if self._block_args.expand_ratio != 1:
            self._expand_conv = conv1x1(inp, oup)

            self._bn0 = nn.ModuleList([nn.BatchNorm2d((i+1)*self._expan_channel_per_subnet,
                                                      momentum=self._bn_mom,
                                                      eps=self._bn_eps) for i in range(width)])

        # Depthwise convolution
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = get_same_padding_conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k,
                                                       stride=s)
        self._bn1 = nn.ModuleList([nn.BatchNorm2d((i+1)*self._expan_channel_per_subnet,
                                                  momentum=self._bn_mom,
                                                  eps=self._bn_eps) for i in range(width)])

        # Squeeze and Excitation
        if self.has_se:
            self._se_reduce = nn.ModuleList()
            self._se_expand = nn.ModuleList()
            num_squeezed_channels = int(self._block_args.input_filters * self._block_args.se_ratio)
            self._squeeze_channels_per_subnet = num_squeezed_channels / width

            for i in range(width):
                squeeze_channel = max(1, int((i+1)*self._squeeze_channels_per_subnet))
                se_reduce = get_same_padding_conv2d(in_channels=(i+1)*self._expan_channel_per_subnet,
                                                    out_channels=squeeze_channel,
                                                    kernel_size=1)
                se_expand = get_same_padding_conv2d(in_channels=squeeze_channel,
                                                    out_channels=(i+1)*self._expan_channel_per_subnet,
                                                    kernel_size=1)
                self._se_reduce.append(se_reduce)
                self._se_expand.append(se_expand)

        # Pointwise convolution
        final_oup = self._block_args.output_filters
        self._final_channel_per_subnet = final_oup // width
        self._project_conv = get_same_padding_conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1)
        self._bn2 = nn.ModuleList([nn.BatchNorm2d((i+1)*self._final_channel_per_subnet,
                                                  momentum=self._bn_mom,
                                                  eps=self._bn_eps) for i in range(width)])
        self._silu = nn.SiLU(inplace=True)
        self.flops_count = 0

    def reset_flops_count(self):
        self.flops_count = 0

    def forward(self, inputs: Tuple, drop_connect_rate=None):
        """
        MBConv forward function
        Args:
            inputs: a tuple of (input_tensor, width)
            drop_connect_rate: drop depth

        Returns:
            out: (output_tensor, width) tuple
        """
        input_tensor, width = inputs
        x = input_tensor
        input_channels = (width+1)*self._input_channel_per_subnet
        expand_channels = (width+1)*self._expan_channel_per_subnet
        output_channels = (width+1)*self._final_channel_per_subnet

        if self._block_args.expand_ratio != 1:
            x = F.conv2d(x, self._expand_conv.weight[:expand_channels, :input_channels])
            x = self._bn0[width](x)
            x = self._silu(x)
            self.flops_count += _count_flops_conv2d(1, input_tensor.shape[3], input_channels, output_channels)\
                            + _count_flops_batchnorm2d(x.numel()) + _count_flops_relu(x.numel())

        self.flops_count += _count_flops_depthwise_conv2d(self._depthwise_conv.kernel_size[0],
                                                          x.shape[3]//self._depthwise_conv.stride[0],
                                                          expand_channels)
        x = F.conv2d(x, self._depthwise_conv.weight[:expand_channels, :expand_channels],
                     padding=self._depthwise_conv.padding, groups=expand_channels,
                     stride=self._depthwise_conv.stride)
        x = self._bn1[width](x)
        x = self._silu(x)
        self.flops_count += _count_flops_batchnorm2d(x.numel()) + _count_flops_relu(x.numel())

        # Squeeze and Excitation
        if self.has_se:
            x_squeeze = F.adaptive_avg_pool2d(x, 1)
            self.flops_count += _count_flops_conv2d(self._se_reduce[width].kernel_size[0],
                                                    x.shape[3],
                                                    self._se_reduce[width].in_channels,
                                                    self._se_reduce[width].out_channels)
            x_squeeze = self._se_reduce[width](x_squeeze)
            x_squeeze = self._silu(x_squeeze)
            self.flops_count += _count_flops_conv2d(self._se_expand[width].kernel_size[0],
                                                    x.shape[3],
                                                    self._se_expand[width].in_channels,
                                                    self._se_expand[width].out_channels)
            self.flops_count += _count_flops_relu(x_squeeze.numel())
            x_squeeze = self._se_expand[width](x_squeeze)
            x = torch.sigmoid(x_squeeze) * x
            self.flops_count += _count_flops_relu(x.numel())

        # Pointwise Convolution
        self.flops_count += _count_flops_conv2d(1, x.shape[3], expand_channels, output_channels)
        x = F.conv2d(x, self._project_conv.weight[:output_channels, :expand_channels],
                     padding=self._project_conv.padding)
        x = self._bn2[width](x)
        self.flops_count += _count_flops_batchnorm2d(x.numel())

        # Skip connection and drop connect
        if self._id_skip and self._block_args.stride == 1 and input_channels == output_channels:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)

            x = x + input_tensor
            self.flops_count += x.numel()

        return x, width


class AnytimeEfficientNet(nn.Module):
    """ Anytime EfficientNet-B0 model """
    def __init__(self, block_args: List[BlockArgs],
                 global_params: GlobalParams, width: int = 1, depth: int = 1):
        super(AnytimeEfficientNet, self).__init__()
        self._global_params = global_params
        self._block_args = block_args
        self._width = width
        self._depth = depth
        self.exit_stages = [2, 3, 4]
        self.depth_exits = []
        self.width_exits = list(range(width))

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3
        out_channels = 32
        self._stem_channel_per_subnet = out_channels // self._width
        self._conv_stem = get_same_padding_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                                  stride=2)
        self._bn0 = nn.ModuleList([nn.BatchNorm2d((i + 1) * self._stem_channel_per_subnet,
                                                  momentum=bn_mom,
                                                  eps=bn_eps) for i in range(width)])

        # Build blocks
        self._blocks = nn.ModuleList([])
        self.output_layers = nn.ModuleList([])

        for i, block_args in enumerate(self._block_args):
            self._blocks.append(AnytimeMBConvBlock(block_args, global_params, self._width))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)

            for _ in range(block_args.num_repeat-1):
                self._blocks.append(AnytimeMBConvBlock(block_args, global_params, width=self._width))

            if i in self.exit_stages:
                channel_per_subnetwork = block_args.output_filters // self._width
                output_subnetworks = nn.ModuleList([nn.Linear((w+1)*channel_per_subnetwork,
                                                              self._global_params.num_classes)
                                                    for w in range(self._width)])
                self.output_layers.append(output_subnetworks)
                self.depth_exits.append(len(self._blocks))

        # Head
        in_channels = block_args.output_filters
        out_channels = 1280
        self._head_in_channel_per_subnetwork = in_channels // self._width
        self._head_channel_per_subnetwork = out_channels // self._width
        self._conv_head = get_same_padding_conv2d(in_channels, out_channels, kernel_size=1)
        self._bn1 = nn.ModuleList([nn.BatchNorm2d((i + 1) * self._head_channel_per_subnetwork,
                                                  momentum=bn_mom,
                                                  eps=bn_eps) for i in range(width)])

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.ModuleList([nn.Linear((i+1)*self._head_channel_per_subnetwork, self._global_params.num_classes)
                                  for i in range(width)])
        self.output_layers.append(self._fc)
        self._silu = nn.SiLU(inplace=True)

        self.depth_exits.append(len(self._blocks))
        self.flops_count = 0

    def reset_flops_count(self):
        for block in self._blocks:
            block.reset_flops_count
        self.flops_count = 0

    def forward(self, inputs: torch.Tensor, width: int, depth: int) -> torch.Tensor:
        """
        Forward function
        Args:
            inputs: input tensor
            width: width of the sub-network
            depth: depth of the sub-network

        Returns:

        """
        exit_depth = self.depth_exits[depth]

        # Stem
        self.flops_count += _count_flops_conv2d(self._conv_stem.kernel_size[0],
                                                inputs.shape[3]//2,
                                                3, (width+1)*self._stem_channel_per_subnet)
        x = F.conv2d(inputs, self._conv_stem.weight[:(width+1)*self._stem_channel_per_subnet],
                     stride=self._conv_stem.stride, padding=self._conv_stem.padding)
        x = self._bn0[width](x)
        x = self._silu(x)
        self.flops_count += _count_flops_batchnorm2d(x.numel()) + _count_flops_relu(x.numel())

        # Blocks
        for idx in range(exit_depth):
            block = self._blocks[idx]
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)

            x, _ = block((x, width), drop_connect_rate=drop_connect_rate)
            self.flops_count += block.flops_count

        if exit_depth == len(self._blocks):
            # Head
            self.flops_count += _count_flops_conv2d(self._conv_head.kernel_size[0],
                                                    x.shape[3],
                                                    (width + 1) * self._head_in_channel_per_subnetwork,
                                                    (width + 1) * self._head_channel_per_subnetwork)
            x = F.conv2d(x, self._conv_head.weight[:(width+1)*self._head_channel_per_subnetwork,
                                                   :(width+1)*self._head_in_channel_per_subnetwork])
            x = self._bn1[width](x)
            x = self._silu(x)
            self.flops_count += _count_flops_batchnorm2d(x.numel()) + _count_flops_relu(x.numel())

        # Pooling
        x = self._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self.output_layers[depth][width](x)
        self.flops_count += _count_flops_linear(self.output_layers[depth][width].in_features,
                                                self.output_layers[depth][width].out_features)

        return x


def anytime_efficientnet_b0(num_classes: int, pretrained: bool = True) -> AnytimeEfficientNetDepth:
    """ Get anytime efficientnet-b0 model """
    global_params = get_global_param("efficientnet-b0")
    global_params = global_params._replace(num_classes=num_classes)
    model = AnytimeEfficientNetDepth(EFFICIENTNET_B0_BLOCK_ARGS, global_params)

    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["efficientnet-b0"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model


def anytime_efficientnet_b1(num_classes: int, pretrained: bool = True) -> AnytimeEfficientNetDepth:
    """ Get anytime efficientnet-b0 model """
    global_params = get_global_param("efficientnet-b1")
    global_params = global_params._replace(num_classes=num_classes)
    model = AnytimeEfficientNetDepth(EFFICIENTNET_B0_BLOCK_ARGS, global_params)

    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["efficientnet-b1"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model


def anytime_efficientnet_b2(num_classes: int, pretrained: bool = True) -> AnytimeEfficientNetDepth:
    """ Get anytime efficientnet-b0 model """
    global_params = get_global_param("efficientnet-b2")
    global_params = global_params._replace(num_classes=num_classes)
    model = AnytimeEfficientNetDepth(EFFICIENTNET_B0_BLOCK_ARGS, global_params)

    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["efficientnet-b2"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model
