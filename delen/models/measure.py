#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _count_flops_conv2d(k: int, r: int, inplanes: int, planes: int) -> int:
    """
    Return the FLOPS for Conv2d layer
    Reference: https://www.math.purdue.edu/~nwinovic/deep_learning.html
    Args:
        k: kernel size
        r: input resolution
        inplanes: number of input channels
        planes: number of output channels

    Returns:
        flops
    """
    return 1 * k * k * r * r * inplanes * planes


def _count_flops_depthwise_conv2d(k: int, r: int, inplanes: int) -> int:
    """
    Return the FLOPS of Depthwise Conv2d layer
    Args:
        k: kernel size
        r: resolution
        inplanes: input channels
        planes: output channels

    Returns:
        flops

    """
    return inplanes * r * r * k * k


def _count_flops_batchnorm2d(num_elements: int) -> int:
    """
    Return the FLOPS for batchnorm layer
    Args:
        num_elements: number of elements of input tensor

    Returns:

    """
    return 2 * num_elements


def _count_flops_relu(num_elements: int) -> int:
    """
    Return the FLOPS for ReLU layer
    Args:
        num_elements: number of elements of input tensor

    Returns:

    """
    return num_elements


def _count_flops_linear(inplnaes: int, planes: int) -> int:
    """
    Return FLOPS for Linear layer
    Args:
        inplnaes: input channel
        planes: output channel

    Returns:

    """
    return 2 * inplnaes * planes