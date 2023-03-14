#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pycuda.driver as cuda
import numpy as np
import timeit
import tensorrt as trt
from collections import namedtuple

HostDeviceMemory = namedtuple("HostDeviceMemory", "host device")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def allocate_buffers(engine, batch_size, stream=None, input_buf=None):
    """
    Create memory buffer for both host and device
    Args:
        engine: TensorRT engine

    Returns:

    """
    inputs = []
    outputs = []
    bindings = []
    output_shapes = []
    stream = cuda.Stream() if not stream else stream

    print("Acclocating buffer...")
    print("Device memory needed: %d" % engine.device_memory_size)
    print("Current Batch size: %d" % batch_size)

    for binding in engine:
        if engine.binding_is_input(binding) and input_buf:
            inputs = input_buf
            bindings.append(int(input_buf[0].device))
        else:
            input_shape = engine.get_binding_shape(binding)
            input_shape[0] = batch_size
            print("Binding: %s, shape: " % binding, input_shape)
            mem_size = trt.volume(input_shape)
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(mem_size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))

            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_mem, device_mem))
            else:
                output_shapes.append((binding, input_shape))
                outputs.append(HostDeviceMemory(host_mem, device_mem))

    return inputs, outputs, bindings, stream, output_shapes


def do_inference(context, bindings, inputs, outputs, stream, input_shape):
    """
    Perform memcpy_h2d -> launch kernels -> memcpy_d2h -> synchronize
    Args:
        context: CUDA context
        bindings: bindings
        inputs: input buffer
        outputs: output buffer
        stream: stream

    Returns:
        output: host memory buffer

    """
    # Memcpy h2d
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Launch kernels
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Memcpy d2h
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchroinze
    stream.synchronize()
    # Return host mem buffer
    return [out.host for out in outputs]


def build_engine(onnx_path, engine_name, engine_path, shape, save):
    """
    This is the function to create the TensorRT engine
    Args:
       onnx_path : Path to onnx_file.
       shape : Shape of the input of the ONNX file.
       engine_path: Path o output engine file
    """
    print("Start Building Engine ", engine_name, " with Batch ", shape[0])

    with trt.Builder(TRT_LOGGER) as builder,\
            builder.create_network(1) as network,\
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path+engine_name, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        if save:
            with open(engine_path, "wb") as f:
                f.write(engine.serialize())
        print("End Building Engine ", engine_name, " with Batch ", shape[0])
        return engine