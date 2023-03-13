#!/usr/bin/env python3
"""
    Created date: 3/13/23
"""

import os
import argparse
import csv
import yaml
import time
import functools
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from delen.utils import datasets
from delen.models import anytime_resnet50, anytime_resnet18, anytime_resnet34, anytime_efficientnet_b0, \
    anytime_efficientnet_b1, anytime_efficientnet_b2, anytime_wav2vec2

from typing import Tuple, List, Callable, Union
from transformers.models.wav2vec2 import Wav2Vec2Processor

print = functools.partial(print, flush=True)

PRINT_PER_ITER = 100

DATASET_DIRS = {
    "food-101"       : "dataset/food-101",
    "speech_commands": "dataset/speech_commands"
}

DATASETS = {
    "food-101"       : datasets.Food101,
    "speech_commands": datasets.SpeechCommandsDataset
}


def get_first_timestamp(inputs: torch.Tensor):
    """ Given a tensor output in shape [b, t, d], return only the first timestamp """
    return inputs[:, 0, :]


def weighted_loss(model: torch.nn.Module,
                  inputs: torch.Tensor,
                  labels: torch.Tensor,
                  criterion: Callable,
                  optimizer: optim.Optimizer,
                  weights: np.ndarray,
                  postprocess: Callable = None) -> Tuple[torch.Tensor, np.ndarray]:
    """ Perform weighted loss SGD and return the loss """
    loss = torch.tensor(0.)
    loss = loss.cuda() if torch.cuda.is_available() else loss

    optimizer.zero_grad()

    outputs = model(inputs)
    if postprocess:
        outputs = [postprocess(out) for out in outputs]

    for w, out in zip(weights, outputs):
        loss += w * criterion(out, labels)

    accuracies = np.zeros([len(outputs)])
    for j, oup in enumerate(outputs):
        predicted = torch.argmax(oup, 1)
        accuracies[j] += (predicted == labels).sum().cpu().numpy()
    accuracies = accuracies / inputs.shape[0]

    loss.backward()
    return loss, accuracies


def residual_loss(model: torch.nn.Module,
                  inputs: torch.Tensor,
                  labels: torch.Tensor,
                  criterion: Callable,
                  optimizer: optim.Optimizer,
                  postprocess: Callable = None) -> Tuple[torch.Tensor, np.ndarray]:
    """ Perform weighted loss SGD and return the loss """
    optimizer.zero_grad()

    outputs = model(inputs)
    if postprocess:
        outputs = [postprocess(out) for out in outputs]

    output_sum = outputs[0]

    for oup in outputs[1:]:
        output_sum += oup

    loss = criterion(output_sum, labels)
    loss.backward()

    accuracies = np.zeros([len(outputs)])
    for j, oup in enumerate(outputs):
        predicted = torch.argmax(oup, 1)
        accuracies[j] += (predicted == labels).sum().cpu().numpy()
    accuracies = accuracies / inputs.shape[0]

    return loss, accuracies


def eval_accuracy_indep(model: torch.nn.Module,
                        test_loader: torch.utils.data.DataLoader,
                        num_subnetworks: int,
                        gpu: bool,
                        postprocess: Callable = None) -> np.ndarray:
    """ Return the accuracy of all sub-networks, the output digits are independent """
    accuracies = np.zeros([num_subnetworks])
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i % PRINT_PER_ITER == 0:
                print("Evaluation iter {:d}/{:d}".format(i, len(test_loader)))
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            if postprocess:
                outputs = [postprocess(out) for out in outputs]

            for j, oup in enumerate(outputs):
                predicted = torch.argmax(oup, 1)
                accuracies[j] += (predicted == labels).sum().cpu().numpy()

    accuracies = accuracies / (len(test_loader)*test_loader.batch_size)
    return accuracies


def eval_accuracy_residual(model: torch.nn.Module,
                           test_loader: torch.utils.data.DataLoader,
                           num_subnetworks: int,
                           gpu: bool,
                           postprocess: Callable = None) -> np.ndarray:
    """
    Return the accuracy of all sub-networks, the output digits of exit i is the sum
    of all previous output digits
    """
    accuracies = np.zeros([num_subnetworks])
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i % PRINT_PER_ITER == 0:
                print("Evaluation iter {:d}/{:d}".format(i, len(test_loader)))
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            if postprocess:
                outputs = [postprocess(out) for out in outputs]

            output_sum = torch.zeros_like(outputs[0])
            for j, oup in enumerate(outputs):
                output_sum += oup
                predicted = torch.argmax(output_sum, 1)
                accuracies[j] += (predicted == labels).sum().cpu().numpy()

    accuracies = accuracies / (len(test_loader)*test_loader.batch_size)
    return accuracies


def train(train_loader: torch.utils.data.DataLoader,
          valid_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          model: torch.nn.Module,
          optimizer: optim.Optimizer,
          lr: float,
          loss_fn: Callable,
          eval_fn : Callable,
          num_epoch: int = 50,
          output_dir: str = "./",
          gpu: bool = True,
          one_cycle: bool = True) -> Tuple[torch.nn.Module, np.ndarray, List, List]:
    """ train Anytime model """
    # Load model
    if gpu:
        model = model.cuda()

    checkpoint_file = os.path.join(output_dir, "checkpoint.pth")

    if one_cycle:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*25, epochs=num_epoch,
                                                  steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()

    validation_results = []
    training_data = []
    for epoch in range(num_epoch):
        model = model.train()
        for iteration, (inputs, labels) in enumerate(train_loader):
            start_t = time.time()
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            loss, acc = loss_fn(model, inputs, labels, criterion, optimizer)

            # Save training data
            optimizer.step()

            if one_cycle:
                scheduler.step()

            training_data.append({
                "iteration" : epoch * len(train_loader) + iteration,
                "train_loss": float(loss.detach().cpu().numpy())
            })
            duration = time.time() - start_t
            if iteration % PRINT_PER_ITER == 0:
                print("Training iter {:d}/{:d}, train loss: {:.2f}, time: {:.2f} s/iter"
                      .format(iteration, len(train_loader), training_data[-1]["train_loss"], duration))
                print("Train acc:")
                print(acc)

        print('')

        print("[INFO] Saving checkpoint to {:s}".format(checkpoint_file))
        torch.save(model.state_dict(), checkpoint_file)

        model = model.eval()
        accuracies = eval_fn(model, valid_loader, gpu=gpu)
        epoch_stat = dict(epoch=epoch)
        for i, acc in enumerate(accuracies):
            epoch_stat["exit%d" % i] = acc
        validation_results.append(epoch_stat)

        print("[VALID] Epoch {:d}, validation accuracies: ".format(epoch))
        print(accuracies)
        accuracies = eval_fn(model, test_loader, gpu=gpu)
        print("[TEST] Epoch {:d}, test accuracies: ".format(epoch))
        print(accuracies)

    model = model.eval()
    accuracies = eval_fn(model, test_loader, gpu=gpu)
    print("[TEST] Final test accuracies:")
    print(accuracies)
    return model, accuracies, training_data, validation_results


def get_preprocessor(input_type: str, dataset_type: str) -> Callable:
    """
    Return a callable preprocessor for specific input type and dataset type
    Args:
        input_type: image | audio
        dataset_type: train | test

    Returns:
        preprocessor
    """
    if input_type == "image":
        if dataset_type == "train":
            preprocessor = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.2, hue=.1),
                transforms.RandomAffine(15),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        elif dataset_type == "test":
            preprocessor = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            raise ValueError("Unsupported dataset type {:s}".format(dataset_type))
    elif input_type == "audio":
        preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    else:
        raise ValueError("Unsupported input type {:s}".format(input_type))

    return preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", default=32, type=int, dest="batch_size", help="Training batch size")
    parser.add_argument("-l", "--learning-rate", default=1e-3, dest="lr", type=float, help="learning rate")
    parser.add_argument("--optimizer", default="sgd", dest="optimizer", type=str, help="optimizer")
    parser.add_argument("--eval-fn", default="independent", dest="eval_fn", type=str, help="evaluate function")
    parser.add_argument("--loss-fn", default="weighted_sum", dest="loss_fn", type=str, help="loss function")
    parser.add_argument("-n", "--epoch", default=50, type=int, dest="epoch", help="number of epoch")
    parser.add_argument("-o", "--output-dir", type=str,
                        default="tmp/train/".format(int(time.time())),
                        dest="output_dir",
                        help="output directory")
    parser.add_argument("--dataset", default="food-101", type=str, dest="dataset", help="training dataset")
    parser.add_argument("-m", "--model", required=True, dest="model", type=str, help="train model name")
    parser.add_argument('--weights', type=float, nargs='+', default=[], help="loss weight of sub-networks")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="if load pretrained parameters")
    parser.add_argument("--input-type", type=str, required=True, dest="input_type",
                        help="Input type (e.g. image | audio)")
    parser.add_argument("--one-cycle",  dest="one_cycle", action="store_true",
                        help="if use one cycle scheduler")
    args = parser.parse_args()

    logging.basicConfig(format="[%(levelname)s] %(funcName)s %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)

    models_set = {
        "resnet18"       : anytime_resnet18,
        "resnet34"       : anytime_resnet34,
        "resnet50"       : anytime_resnet50,
        "efficientnet-b0": anytime_efficientnet_b0,
        "efficientnet-b1": anytime_efficientnet_b1,
        "efficientnet-b2": anytime_efficientnet_b2,
        "wav2vec2"       : anytime_wav2vec2
    }

    postprocess_fn = {
        "wav2vec2": get_first_timestamp
    }

    optimizer_set = {
        "sgd" : functools.partial(optim.SGD, momentum=0.9, weight_decay=0),
        "adam": optim.Adam
    }

    loss_fn_set = {
        "weighted_sum": weighted_loss,
        "residual"    : residual_loss,
    }

    eval_fn_set = {
        "independent": eval_accuracy_indep,
        "residual"   : eval_accuracy_residual
    }

    train_transform = get_preprocessor(args.input_type, "train")
    test_transform = get_preprocessor(args.input_type, "test")

    train_dataset = DATASETS[args.dataset](DATASET_DIRS[args.dataset], transform=train_transform, mode="train")
    valid_dataset = DATASETS[args.dataset](DATASET_DIRS[args.dataset], transform=test_transform, mode="valid")
    test_dataset = DATASETS[args.dataset](DATASET_DIRS[args.dataset], transform=test_transform, mode="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    logger.info("Training {:s}".format(args.model))
    model = models_set[args.model](num_classes=train_dataset.num_classes, pretrained=args.pretrained)
    num_subnetworks = model.num_subnetworks

    if (n_gpu := torch.cuda.device_count()) > 1:
        logger.info("Using {:d} GPUs".format(n_gpu))
        model = nn.DataParallel(model)

    if not args.weights and args.loss_fn == "weighted_sum":
        logger.warning("No weight specified, using even weights")
        weights = np.ones([num_subnetworks])
        weights = weights / weights.size
    else:
        weights = np.array(args.weights)

    logger.info("Using {:s} optimizer, with learning rate: {:f}".format(args.optimizer, args.lr))
    optimizer = optimizer_set[args.optimizer](model.parameters(), lr=args.lr)

    logger.info("Using {:s} as loss function, {:s} as evaluation method".format(args.loss_fn, args.eval_fn))
    loss_fn = loss_fn_set[args.loss_fn]
    if args.loss_fn == "weighted_sum":
        loss_fn = functools.partial(loss_fn, weights=weights)

    post_fn = postprocess_fn[args.model] if args.model in postprocess_fn else None
    loss_fn = functools.partial(loss_fn, postprocess=post_fn)
    eval_fn = functools.partial(eval_fn_set[args.eval_fn], num_subnetworks=num_subnetworks, postprocess=post_fn)

    logger.info("Using one cycle scheduler: {:s}".format(str(args.one_cycle)))
    trained_model, accuracies, training_data, validation_results = train(train_loader,
                                                                         valid_loader,
                                                                         test_loader,
                                                                         model,
                                                                         optimizer,
                                                                         args.lr,
                                                                         loss_fn,
                                                                         eval_fn,
                                                                         args.epoch,
                                                                         output_dir,
                                                                         torch.cuda.is_available(),
                                                                         one_cycle=args.one_cycle)

    weight_path = os.path.join(output_dir, "weights.pth")
    torch.save(trained_model.state_dict(), weight_path)
    logger.info("Training finished, output weights are saved to %s" % weight_path)

    record = dict(learning_rate=args.lr, batch_size=args.batch_size,
                  num_epoch=args.epoch, loss_weights=[float(w) for w in weights.flatten()],
                  accuracies=[float(acc) for acc in accuracies.flatten()])

    spec_filename = os.path.join(output_dir, "train_spec.yml")
    with open(spec_filename, 'w') as f:
        yaml.dump(record, f)
    logger.info("Training specs are saved to %s" % spec_filename)

    training_data_filename = os.path.join(output_dir, "training_trace.csv")
    with open(training_data_filename, 'w') as f:
        writer = csv.DictWriter(f, training_data[0].keys())
        writer.writeheader()
        writer.writerows(training_data)
    logger.info("Training loss trace is saved to {:s}".format(training_data_filename))

    validation_data_filename = os.path.join(output_dir, "validation_results.csv")
    with open(validation_data_filename, 'w') as f:
        writer = csv.DictWriter(f, validation_results[0].keys())
        writer.writeheader()
        writer.writerows(validation_results)
    logger.info("Validation data is saved to {:s}".format(validation_data_filename))


if __name__ == '__main__':
    main()