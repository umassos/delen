#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch

from collections import OrderedDict
from delen.models import anytime_resnet50, anytime_resnet18, anytime_resnet34, anytime_efficientnet_b0, \
    anytime_efficientnet_b1,anytime_efficientnet_b2, anytime_wav2vec2

models_set = {
    "resnet18"       : anytime_resnet18,
    "resnet34"       : anytime_resnet34,
    "resnet50"       : anytime_resnet50,
    "efficientnet-b0": anytime_efficientnet_b0,
    "efficientnet-b1": anytime_efficientnet_b1,
    "efficientnet-b2": anytime_efficientnet_b2,
    "wav2vec2"       : anytime_wav2vec2
}


def main() -> None:
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("-m", "--model", required=True, type=str, dest="model", help="model name")
    parser.add_argument("-c", "--num-classes", required=True, type=int, dest="num_classes", help="number of classes")
    parser.add_argument("-d", "--model-dir", required=True, type=str, dest="model_dir", help="model directory")
    parser.add_argument("-g", "--granularity", default="exit", type=str,
                        dest="granularity", help="granularity of sub-network")
    parser.add_argument("-o", "--output-dir", default="subnetworks", dest="output_dir", help="output directory")
    parser.add_argument("--input-size", dest="input_size", nargs='+', type=int, help="Input size")
    args = parser.parse_args()

    output_dir = os.path.join(args.model_dir, args.output_dir)
    weight_path = os.path.join(args.model_dir, "weights.pth")
    os.makedirs(output_dir, exist_ok=True)

    model = models_set[args.model](num_classes=args.num_classes, pretrained=False)
    weights = torch.load(weight_path, map_location=torch.device('cpu'))

    # Model trained with nn.DataParallel has weights begin with `module.`
    # fixed_weights = OrderedDict()
    # for name, weight in weights.items():
    #     fixed_weights[name[7:]] = weight

    model.load_state_dict(weights)
    model = model.eval()

    if args.granularity == "exit":
        subnetworks = [model.get_subnetwork(i) for i in range(model.num_subnetworks)]
    elif args.granularity == "block":
        subnetworks = [model.get_block(i) for i in range(model.num_blocks)]
    elif args.granularity == "model":
        subnetworks = [model.restore_regular_model()]
    else:
        raise ValueError("Granularity type {:s} is not supported".format(args.granularity))

    input_size = args.input_size if args.input_size else [1, 3, 224, 224]
    inputs = torch.rand(input_size)

    x = inputs

    for i, net in enumerate(subnetworks):
        input_names = ["input"]
        output_names = ["output"]

        if i < len(subnetworks) - 1:
            if args.granularity == "exit":
                output_names.append("features")
            else:
                output_names[0] = "features"

        model_onnx_path = os.path.join(output_dir, args.model + "_net{:02d}.onnx".format(i))
        print("Exporting {:s}".format(model_onnx_path))
        torch.onnx.export(net,
                          x,
                          model_onnx_path,
                          export_params=True,
                          do_constant_folding=True,
                          input_names=input_names,
                          output_names=output_names)

        if i < len(subnetworks) - 1:
            if args.granularity == "exit":
                _, x = net(x)
            else:
                x = net(x)

    print("Done")


if __name__ == '__main__':
    main()