#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import conv1x1, conv3x3
from .measure import _count_flops_conv2d, _count_flops_batchnorm2d, _count_flops_relu, _count_flops_linear
from .anytime_common import AnytimeSubNetwork
from torch.hub import load_state_dict_from_url
from typing import List, Tuple, Union, Callable, Type

MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: bool = False,
            norm_layer: Callable = nn.BatchNorm2d
    ) -> None:
        super(BasicBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.stride = stride
        self.downsample = None

        if downsample:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, stride=stride),
                                            norm_layer(self.expansion * self.planes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Resnet bottleneck architecture
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: bool = False,
                 norm_layer: nn.Module = nn.BatchNorm2d
                 ) -> None:
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = None

        self.inplanes = inplanes
        self.planes = planes
        self.bn1 = norm_layer(self.planes)
        self.bn2 = norm_layer(self.planes)
        self.bn3 = norm_layer(self.expansion * self.planes)

        if downsample:
            self.downsample = nn.Sequential(conv1x1(inplanes, planes * self.expansion, stride=stride),
                                            norm_layer(self.expansion * self.planes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Process x using k sub-network
        Args:
            inputs: a tuple of (x, k) where x is the input tensor and k is the width

        Returns:
            Output tensor

        """
        x = inputs
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class AnytimeBottleneckWidthDepth(nn.Module):
    # Resnet bottleneck architecture with width nesting
    expansion = 4

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 num_subnets: int,
                 stride: int = 1,
                 downsample: bool = False,
                 norm_layer: nn.Module = nn.BatchNorm2d
                 ) -> None:
        super(AnytimeBottleneckWidthDepth, self).__init__()
        assert planes % num_subnets == 0, \
            "The number of sub-networks should be a divisor of planes, got {:d} and {:d}" \
                .format(num_subnets, planes)
        assert inplanes % num_subnets == 0, \
            "The number of sub-networks should be a divisor of inplanes, got {:d} and {:d}" \
                .format(num_subnets, planes)

        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.num_subnets = num_subnets
        self.downsample = None
        self.downsample_bn = nn.ModuleList()

        self.inplanes_per_subnet = inplanes // num_subnets
        self.planes_per_subnet = planes // num_subnets
        self.bn1 = nn.ModuleList([norm_layer((i + 1) * self.planes_per_subnet) for i in range(num_subnets)])
        self.bn2 = nn.ModuleList([norm_layer((i + 1) * self.planes_per_subnet) for i in range(num_subnets)])
        self.bn3 = nn.ModuleList(
                [norm_layer((i + 1) * self.expansion * self.planes_per_subnet) for i in range(num_subnets)])

        if downsample:
            self.downsample = conv1x1(inplanes, planes * self.expansion, stride=stride)
            for i in range(num_subnets):
                self.downsample_bn.append(norm_layer((i + 1) * self.expansion * self.planes_per_subnet))

        self.flops_count = 0

    def reset_flops_count(self):
        self.flops_count = 0

    def num_params(self, k) -> int:
        """
        Get the numer of parameters for k-th subnets
        Args:
            k: subnet width

        Returns:
            params: The number of parameters
        """
        params = 0
        params += self.conv1.weight[:(k + 1) * self.planes_per_subnet, :(k + 1) * self.inplanes_per_subnet].numel()
        params += self.bn1[k].weight.numel() + self.bn1[k].bias.numel()
        params += self.conv2.weight[:(k + 1) * self.planes_per_subnet, :(k + 1) * self.planes_per_subnet].numel()
        params += self.bn2[k].weight.numel() + self.bn2[k].bias.numel()
        params += self.conv3.weight[:(k + 1) * self.expansion * self.planes_per_subnet,
                  :(k + 1) * self.planes_per_subnet].numel()
        params += self.bn3[k].weight.numel() + self.bn3[k].bias.numel()

        if self.downsample is not None:
            params += self.downsample.weight[:(k + 1) * self.expansion * self.planes_per_subnet,
                      :(k + 1) * self.inplanes_per_subnet].numel()
            params += self.downsample_bn[k].weight.numel() + self.downsample_bn[k].bias.numel()

        return params

    def forward(self, inputs: tuple) -> Tuple[torch.Tensor, int]:
        """
        Process x using k sub-network
        Args:
            inputs: a tuple of (x, k) where x is the input tensor and k is the width

        Returns:
            Output tensor

        """
        x, k = inputs
        identity = x

        inplanes = (k + 1) * self.inplanes_per_subnet
        planes = (k + 1) * self.planes_per_subnet
        r = x.size(2)

        out = F.conv2d(x, self.conv1.weight[:planes, :inplanes])
        out = self.bn1[k](out)
        out = self.relu(out)
        self.flops_count += _count_flops_conv2d(1, r, inplanes, planes) + _count_flops_batchnorm2d(out.numel()) + \
                            _count_flops_relu(out.numel())

        out = F.conv2d(out, self.conv2.weight[:planes, :planes], padding=1, stride=self.stride)
        out = self.bn2[k](out)
        out = self.relu(out)
        self.flops_count += _count_flops_conv2d(3, r // self.stride, planes, planes) + \
                            _count_flops_batchnorm2d(out.numel()) + _count_flops_relu(out.numel())

        out = F.conv2d(out, self.conv3.weight[:self.expansion * planes, :planes])
        out = self.bn3[k](out)
        self.flops_count += _count_flops_conv2d(1, r // self.stride, planes, self.expansion * planes) + \
                            _count_flops_batchnorm2d(out.numel()) + _count_flops_relu(out.numel())

        if self.downsample is not None:
            identity = F.conv2d(identity, self.downsample.weight[:self.expansion * planes, :inplanes],
                                stride=self.stride)
            identity = self.downsample_bn[k](identity)
            self.flops_count += _count_flops_conv2d(1, r // self.stride, inplanes, self.expansion * planes) + \
                                _count_flops_batchnorm2d(identity.numel())

        out += identity
        out = self.relu(out)
        self.flops_count += _count_flops_relu(out.numel()) + out.numel()

        return out, k


class AnytimeResnetDepth(nn.Module):
    # Anytime implementation of Resnet
    def __init__(self,
                 layers: List[int],
                 block: Type[Union[BasicBlock, Bottleneck]],
                 num_classes: int = 100,
                 base_width: int = 64,
                 first_stride: int = 2,
                 norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super(AnytimeResnetDepth, self).__init__()

        self.layers = layers
        self.num_classes = num_classes
        self.base_width = base_width
        self.first_stride = first_stride
        self.norm_layer = norm_layer
        self.inplanes = base_width
        self.block = block
        self._blocks = []

        self.conv1 = nn.Conv2d(3, self.base_width, kernel_size=(7, 7), stride=(first_stride, first_stride),
                               padding=(3, 3), bias=False)
        self.bn1 = norm_layer(self.base_width)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])

        self.output_layer1 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Flatten(1),
                                           nn.Linear(self.inplanes, self.num_classes))

        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.output_layer2 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Flatten(1),
                                           nn.Linear(self.inplanes, self.num_classes))

        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.output_layer3 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Flatten(1),
                                           nn.Linear(self.inplanes, self.num_classes))

        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        self.output_layer4 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Flatten(1),
                                           nn.Linear(self.inplanes, self.num_classes))

        # self.bottleneck_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.output_layers = [self.output_layer1, self.output_layer2, self.output_layer3, self.output_layer4]
        self.sub_networks = [
            nn.Sequential(self.conv1, self.bn1, self.relu, self.max_pool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4
        ]
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def num_subnetworks(self):
        return len(self.sub_networks)

    @property
    def num_blocks(self):
        return len(self._blocks)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        """ Make Resnet layer """
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = list()
        cur_block = block(self.inplanes, planes, stride, downsample, self.norm_layer)
        layers.append(cur_block)
        self._blocks.append(cur_block)
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            cur_block = block(self.inplanes, planes)
            layers.append(cur_block)
            self._blocks.append(cur_block)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple:
        """ Inference with width and depth """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        out1 = self.output_layer1(x)

        x = self.layer2(x)
        out2 = self.output_layer2(x)

        x = self.layer3(x)
        out3 = self.output_layer3(x)

        x = self.layer4(x)
        x = self.output_layer4(x)

        return out1, out2, out3, x
        # if self.training:
        #     return out1, out2, out3, x
        # else:
        #     return x

    def get_subnetwork(self, idx: int) -> nn.Module:
        """ Return get subnetwork given index """
        if idx < 0 or idx >= self.num_subnetworks:
            raise ValueError("Sub-network index out of range, available sub-networks {:d}, got {:d}".format(
                    self.num_subnetworks, idx))

        include_features = idx < self.num_subnetworks - 1
        return AnytimeSubNetwork(self.sub_networks[idx], self.output_layers[idx], include_features)

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
                    self.conv1,
                    self.bn1,
                    self.relu,
                    self.max_pool,
                    self._blocks[idx]
            )
        elif idx == self.num_blocks - 1:
            return nn.Sequential(
                    self._blocks[idx],
                    self.output_layer4
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
                self.conv1,
                self.bn1,
                self.relu,
                self.max_pool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.output_layer4
        )


class AnytimeResnetWidthDepth(nn.Module):
    # Anytime implementation of Resnet
    def __init__(self,
                 layers: List[int],
                 width_subnets: int,
                 num_classes: int = 100,
                 base_width: int = 64,
                 first_stride: int = 2,
                 norm_layer: nn.Module = nn.BatchNorm2d) -> None:
        super(AnytimeResnetWidthDepth, self).__init__()
        assert base_width % width_subnets == 0

        self.layers = layers
        self.num_classes = num_classes
        self.base_width = base_width
        self.first_stride = first_stride
        self.norm_layer = norm_layer
        self.width_subnets = width_subnets
        self.inplanes = base_width
        self.base_width_per_subnet = base_width // self.width_subnets

        self.conv1 = nn.Conv2d(3, self.base_width, kernel_size=(7, 7), stride=(first_stride, first_stride),
                               padding=(3, 3), bias=False)
        self.bn1 = nn.ModuleList([norm_layer((i + 1) * self.base_width_per_subnet) for i in range(self.width_subnets)])
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(AnytimeBottleneckWidthDepth, 64, self.layers[0])

        self.output_layer1 = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                          nn.Flatten(1),
                                                          nn.Linear((i + 1) * (self.inplanes // width_subnets),
                                                                    self.num_classes))
                                            for i in range(width_subnets)])

        self.layer2 = self._make_layer(AnytimeBottleneckWidthDepth, 128, self.layers[1], stride=1)
        self.output_layer2 = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                          nn.Flatten(1),
                                                          nn.Linear((i + 1) * (self.inplanes // width_subnets),
                                                                    self.num_classes))
                                            for i in range(width_subnets)])

        self.layer3 = self._make_layer(AnytimeBottleneckWidthDepth, 256, self.layers[2], stride=2)
        self.output_layer3 = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                          nn.Flatten(1),
                                                          nn.Linear((i + 1) * (self.inplanes // width_subnets),
                                                                    self.num_classes))
                                            for i in range(width_subnets)])

        self.layer4 = self._make_layer(AnytimeBottleneckWidthDepth, 512, self.layers[3], stride=1)
        self.output_layer4 = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                                          nn.Flatten(1),
                                                          nn.Linear((i + 1) * (self.inplanes // width_subnets),
                                                                    self.num_classes))
                                            for i in range(width_subnets)])

        self.bottleneck_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.output_layers = [self.output_layer1, self.output_layer2, self.output_layer3, self.output_layer4]
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.flops_count = 0

    def _make_layer(self, block: Type[AnytimeBottleneckWidthDepth], planes: int, num_blocks: int, stride: int = 1) -> \
            nn.Sequential:
        """ Make Resnet layer """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes*block.expansion, stride=stride),
            #     self.norm_layer(planes*block.expansion)
            # )

        layers = list()
        layers.append(block(self.inplanes, planes, self.width_subnets, stride, downsample, self.norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.width_subnets))

        return nn.Sequential(*layers)

    def reset_flops_count(self):
        """ Reset FLOPS count """
        for layer in self.bottleneck_layers:
            for block in layer.children():
                block.reset_flops_count()

        self.flops_count = 0

    def num_params(self, width: int, depth: int) -> int:
        """
        Get the number of parameters for subnet with width w and depth d
        Args:
            width: width of the subnet
            depth: depth of the subnet

        Returns:
            params
        """
        params = 0
        params += self.conv1.weight[:(width + 1) * self.base_width_per_subnet].numel()
        params += self.bn1[width].weight.numel() + self.bn1[width].bias.numel()

        for _, layer in zip(range(depth + 1), self.bottleneck_layers):
            for block in layer.children():
                params += block.num_params(width)

        params += self.output_layers[depth][width][-1].weight.numel()
        return params

    def forward(self, x: torch.Tensor, width: int, depth: int) -> torch.Tensor:
        """ Inference with width and depth """
        r = x.size(2)

        x = F.conv2d(x, self.conv1.weight[:(width + 1) * self.base_width_per_subnet], stride=self.first_stride,
                     padding=3)
        x = self.bn1[width](x)
        x = self.relu(x)
        self.flops_count += _count_flops_conv2d(3, r // 2, 3, (width + 1) * self.base_width_per_subnet) + \
                            _count_flops_batchnorm2d(x.numel()) + _count_flops_relu(x.numel())

        x = self.max_pool(x)

        for _, layer in zip(range(depth + 1), self.bottleneck_layers):
            x, _ = layer((x, width))
            for block in layer.children():
                self.flops_count += block.flops_count

        x = self.output_layers[depth][width](x)
        self.flops_count += _count_flops_linear(self.output_layers[depth][width][-1].weight.shape[1],
                                                self.output_layers[depth][width][-1].weight.shape[0])
        return x


def anytime_resnet50(num_classes: int, pretrained: bool = True) -> AnytimeResnetDepth:
    """
    Return an instance of anytime resnet50 model
    Args:
        num_classes: number of classes
        pretrained: If load pretrained parameters

    Returns:
        model
    """
    model = AnytimeResnetDepth([3, 4, 6, 3], Bottleneck, num_classes=num_classes)
    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["resnet50"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model


def anytime_resnet18(num_classes: int, pretrained: bool = True) -> AnytimeResnetDepth:
    """
    return an instance of anytime resnet18 model
    Args:
        num_classes: number of classes
        pretrained: If load pretrained parameters

    Returns:
        model

    """
    model = AnytimeResnetDepth([2, 2, 2, 2], BasicBlock, num_classes=num_classes)
    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["resnet18"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model


def anytime_resnet34(num_classes: int, pretrained: bool = True) -> AnytimeResnetDepth:
    """
    Return an instance of anytime resnet34 model
    Args:
        num_classes: number of classes
        pretrained: If load pretrained parameters

    Returns:
        model

    """
    model = AnytimeResnetDepth([3, 4, 6, 3], BasicBlock, num_classes=num_classes)
    if pretrained:
        logger.info("Using Imagenet pretrained parameters")
        missing_params = model.load_state_dict(load_state_dict_from_url(MODEL_URLS["resnet34"]), strict=False)
        logger.info("Loadding weights status: {:s}".format(str(missing_params)))

    return model
