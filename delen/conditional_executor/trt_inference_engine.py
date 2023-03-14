#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
import glob
import tensorrt as trt
import logging
import pycuda.driver as cuda

from . import common

from collections import namedtuple

from typing import List, Dict

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SubNetwork = namedtuple("SubNetwork", "engine context inputs outputs bindings stream output_shapes")


class TRTAnytimeDNNEngine:
    """[summary]
    TensorRT Inference Engine
    """

    def __init__(self, model_dir: str, intermediate_output_name: str = "features"):
        """[summary]

        Args:
            engine_file ([type]): [description]
            input_shape ([type]): [description]
            intermediate_output_name: binding for intermediate output
        """
        self.model_dir = model_dir
        self.model_engine_filenames = sorted(glob.glob(os.path.join(model_dir, "*.engine")))
        self.num_subnets = len(self.model_engine_filenames)
        self.sub_networks = []
        self.intermediate_output_name = intermediate_output_name

        trt.init_libnvinfer_plugins(None, '')
        logger.info("Logging anytime DNN with {:d} sub-networks".format(self.num_subnets))

        stream = None
        inputs = None
        for i, engine_file in enumerate(self.model_engine_filenames):
            with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())

            for binding in engine:
                # Only works for single input models
                if engine.binding_is_input(binding):
                    self.input_shape = engine.get_binding_shape(binding)

            context = engine.create_execution_context()

            inputs, outputs, bindings, stream, output_shapes = \
                common.allocate_buffers(engine, int(self.input_shape[0]), stream, inputs)

            self.sub_networks.append(SubNetwork(engine, context, inputs, outputs, bindings, stream, output_shapes))

            inputs = None
            for j, (name, _) in enumerate(output_shapes):
                if name == self.intermediate_output_name:
                    inputs = [outputs[j]]

        self.stream = stream

    def execute_subnet(self, idx: int) -> None:
        """ Execute sub_network[idx] """
        self.sub_networks[idx].context.execute_async_v2(bindings=self.sub_networks[idx].bindings,
                                                        stream_handle=self.stream.handle)

    def get_subnet_output(self, idx: int) -> Dict[str, np.ndarray]:
        """ Copy output buf from device to host and return it """
        self._copy_dtoh(idx)
        self.stream.synchronize()

        outputs = {output_meta[0]: out.host for output_meta, out in
                   zip(self.sub_networks[idx].output_shapes, self.sub_networks[idx].outputs)}
        return outputs

    def _copy_dtoh(self, idx: int) -> None:
        """ Copy output of sub_network[idx] to host """
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.sub_networks[idx].outputs]

    def _copy_htod(self, inputs: np.ndarray):
        """ Copy input of the DNN to device """
        self.sub_networks[0].inputs[0].host[:] = inputs.ravel()
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.sub_networks[0].inputs]

    def infer_to(self, inputs: np.ndarray, idx: int):
        """ Run inference to sub_networks[idx]

        Args:
            inputs : input tensor
            idx: index of sub-network

        Returns:
            [type]: [description]
        """

        self._copy_htod(inputs)

        for i in range(idx+1):
            self.execute_subnet(i)

        return self.get_subnet_output(idx)


class TRTInferenceEngine:
    """[summary]
    TensorRT Inference Engine
    """

    def __init__(self, input_file, batch_size=1):
        """[summary]

        Args:
            engine_file ([type]): [description]
            input_shape ([type]): [description]
        """

        with open(input_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.batch_size = batch_size
        self.inputs, self.outputs, self.bindings, self.stream, self.output_shapes = common.allocate_buffers(
            self.engine, self.batch_size)

    def infer(self, image):
        """[summary]

        Args:
            request (Request): [Request Object]
            ignore_result (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        self.inputs[0].host[:] = image.ravel()

        trt_outputs = common.do_inference(
                self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs,
                stream=self.stream, input_shape=image.shape)

        return trt_outputs
