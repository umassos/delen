#!/usr/bin/env python3
"""
    Training datasets
"""

import os
import glob
import json
import soundfile as sf
import random
import logging
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Food101(Dataset):
    """
    Food-101 dataset
    Reference: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
    """
    def __init__(self, root_dir: str, transform: Callable, mode: str):
        super(Food101, self).__init__()
        self._root_dir = root_dir
        self._transform = transform
        self._mode = mode

        if self._mode == "train":
            meta_file = os.path.join(self._root_dir, "meta/train_ds.txt")
            if not os.path.exists(meta_file):
                self._create_validation_dataset(os.path.join(self._root_dir, "meta/train.txt"))

            logger.info("Loading Food101 utils set from {:s}".format(meta_file))

        elif self._mode == "valid":
            meta_file = os.path.join(self._root_dir, "meta/valid_ds.txt")
            if not os.path.exists(meta_file):
                self._create_validation_dataset(os.path.join(self._root_dir, "meta/train.txt"))

            logger.info("Loading Food101 validation set from {:s}".format(meta_file))
        elif self._mode == "test":
            meta_file = os.path.join(self._root_dir, "meta/test.txt")
            logger.info("Loading Food101 test set from {:s}".format(meta_file))
        else:
            raise ValueError("Unsupported mode {:s}".format(self._mode))

        label_file = os.path.join(self._root_dir, "meta/classes.txt")

        with open(label_file, 'r') as f:
            label_names = f.readlines()
            self._label_index = {label.strip(): i for i, label in enumerate(label_names)}

        with open(meta_file, 'r') as f:
            img_filenames = f.readlines()
            self._file_list = [(os.path.join(self._root_dir, "images/" + filename.strip() + ".jpg"),
                               self._label_index[filename.split('/')[0]])
                               for filename in img_filenames]

        logger.info("Dataset size: {:d}".format(len(self._file_list)))

    def __len__(self):
        return len(self._file_list)

    def __getitem__(self, idx):
        img_filename, label = self._file_list[idx]
        img = Image.open(img_filename).convert("RGB")

        if self._transform:
            img = self._transform(img)
        return img, label

    def _create_validation_dataset(self, train_meta_file):
        """ Create validation datset from utils set """
        np.random.seed(404)

        with open(train_meta_file, 'r') as f:
            all_samples = np.array(f.read().splitlines())

        num_valid_samples = int(len(all_samples) * 0.2)
        valid_idx = np.random.choice(range(len(all_samples)), size=num_valid_samples, replace=False)
        valid_filter = np.zeros(all_samples.shape[0], dtype=bool)
        valid_filter[valid_idx] = True

        train_samples = all_samples[~valid_filter]
        valid_samples = all_samples[valid_filter]

        output_dir = os.path.dirname(train_meta_file)
        train_filename = os.path.join(output_dir, "train_ds.txt")
        valid_filename = os.path.join(output_dir, "valid_ds.txt")

        with open(train_filename, 'w') as f:
            for sample in train_samples:
                f.write(f"{sample}\n")

        with open(valid_filename, 'w') as f:
            for sample in valid_samples:
                f.write(f"{sample}\n")


    @property
    def num_classes(self) -> int:
        """ The number of classes in this dataset """
        return len(self._label_index)

    @staticmethod
    def split_train_valid(root_dir: str, train_frac: float = 0.8, seed: int = 404) -> None:
        """ Split the train.txt into train_ds.txt and valid_ds.txt """
        with open(os.path.join(root_dir, "meta/train.json"), 'r') as f:
            data = json.loads(f.read())

        training_set = []
        validation_set = []
        train_size_per_class = int(750 * train_frac)     # dataset has 750 samples per class

        random.seed(seed)
        for _, samples in data.items():
            random.shuffle(samples)
            training_set.extend(samples[:train_size_per_class])
            validation_set.extend(samples[train_size_per_class:])

        training_set.sort()
        validation_set.sort()

        with open(os.path.join(root_dir, "meta/train_ds.txt"), 'w') as f:
            f.writelines("{:s}\n".format(item) for item in training_set)

        with open(os.path.join(root_dir, "meta/valid_ds.txt"), 'w') as f:
            f.writelines("{:s}\n".format(item) for item in validation_set)


class SpeechCommandsDataset(Dataset):
    """
    Google's Speech Commands Dataset
    Homepage: https://arxiv.org/abs/1804.03209
    """
    def __init__(self, root_dir: str, transform: Callable, mode: str):
        super(SpeechCommandsDataset, self).__init__()
        self._root_dir = root_dir if root_dir[-1] == '/' else root_dir + '/'
        self._preprocessor = transform
        self._mode = mode
        self._sample_length = 16000

        if mode == "train":
            filename = os.path.join(self._root_dir, "training_list.txt")

            # Create utils file list
            if not os.path.exists(filename):
                valid_filename = os.path.join(self._root_dir, "validation_list.txt")
                test_filename = os.path.join(self._root_dir, "testing_list.txt")

                with open(valid_filename, 'r') as valid_file, open(test_filename, 'r') as test_file:
                    valid_set = set(valid_file.read().splitlines())
                    test_set = set(test_file.read().splitlines())

                all_files = glob.glob(self._root_dir + "*/*.wav")
                all_files = set([name.split(self._root_dir)[1] for name in all_files])
                train_set = list(all_files - valid_set - test_set)

                logger.info("Creating utils set with {:d} samples".format(len(train_set)))
                with open(filename, 'w') as f:
                    for path in train_set:
                        f.write("{:s}\n".format(path))

            logger.info("Load SpeechCommands utils set from {:s}".format(filename))
        elif mode == "valid":
            filename = os.path.join(self._root_dir, "validation_list.txt")
            logger.info("Load SpeechCommands validation set from {:s}".format(filename))
        elif mode == "test":
            filename = os.path.join(self._root_dir, "testing_list.txt")
            logger.info("Load SpeechCommands test set from {:s}".format(filename))
        else:
            raise ValueError("Unsupported mode {:s}".format(self._mode))

        with open(filename, 'r') as f:
            self._data_filenames = f.read().splitlines()
            self._data_filenames = [os.path.join(self._root_dir, p) for p in self._data_filenames]

        logger.info("Loaded {:d} samples".format(len(self._data_filenames)))

        # load labels
        class_dirs = glob.glob(self._root_dir + "*/")
        labels = sorted([d.split('/')[-2] for d in class_dirs])

        logger.info("Number of labels: {:d}".format(len(labels)))
        self._idx_to_labels = {}
        self._label_to_idx = {}
        for i, label in enumerate(labels):
            self._idx_to_labels[i] = label
            self._label_to_idx[label] = i

    def __len__(self):
        return len(self._data_filenames)

    def __getitem__(self, idx: int) -> Tuple:
        filename = self._data_filenames[idx]
        label = self._label_to_idx[filename.split('/')[-2]]

        wav, freq = sf.read(filename)
        inputs = self._preprocessor(wav, sampling_rate=freq, return_tensors="pt").input_values[0]

        if inputs.shape[-1] > self._sample_length:
            inputs = inputs[:self._sample_length]
        elif inputs.shape[-1] < self._sample_length:
            pad_size = self._sample_length - inputs.shape[-1]
            inputs = F.pad(inputs, (0, pad_size), "constant", 0)

        return inputs, label

    @property
    def num_classes(self):
        return len(self._label_to_idx)

    @property
    def labels(self):
        return list(self._label_to_idx.keys())


class SpeechCommandsPreprocessedDataset(Dataset):
    """
    Preprocessed speech commands dataset
    """
    def __init__(self, root_dir: str, transform: Callable = lambda x: x, mode: str = "test"):
        super(SpeechCommandsPreprocessedDataset, self).__init__()
        self._root_dir = root_dir if root_dir[-1] == '/' else root_dir + '/'
        self._mode = mode
        self._sample_rate = 16000
        self._preprocessor = transform

        if mode == "train":
            filename = os.path.join(self._root_dir, "training_list.txt")

            # Create utils file list
            if not os.path.exists(filename):
                valid_filename = os.path.join(self._root_dir, "validation_list.txt")
                test_filename = os.path.join(self._root_dir, "testing_list.txt")

                with open(valid_filename, 'r') as valid_file, open(test_filename, 'r') as test_file:
                    valid_set = set(valid_file.read().splitlines())
                    test_set = set(test_file.read().splitlines())

                all_files = glob.glob(self._root_dir + "*/*")
                all_files = set([name.split(self._root_dir)[1] for name in all_files])
                train_set = list(all_files - valid_set - test_set)

                logger.info("Creating utils set with {:d} samples".format(len(train_set)))
                with open(filename, 'w') as f:
                    for path in train_set:
                        f.write("{:s}\n".format(path))

            logger.info("Load SpeechCommands utils set from {:s}".format(filename))
        elif mode == "valid":
            filename = os.path.join(self._root_dir, "validation_list.txt")
            logger.info("Load SpeechCommands validation set from {:s}".format(filename))
        elif mode == "test":
            filename = os.path.join(self._root_dir, "testing_list.txt")
            logger.info("Load SpeechCommands test set from {:s}".format(filename))
        else:
            raise ValueError("Unsupported mode {:s}".format(self._mode))

        with open(filename, 'r') as f:
            self._data_filenames = f.read().splitlines()
            self._data_filenames = [os.path.join(self._root_dir, p) for p in self._data_filenames]
        logger.info("Loaded {:d} samples".format(len(self._data_filenames)))

        # load labels
        class_dirs = glob.glob(self._root_dir + "*/")
        labels = sorted([d.split('/')[-2] for d in class_dirs])

        logger.info("Number of labels: {:d}".format(len(labels)))
        self._idx_to_labels = {}
        self._label_to_idx = {}
        for i, label in enumerate(labels):
            self._idx_to_labels[i] = label
            self._label_to_idx[label] = i

    def __len__(self):
        return len(self._data_filenames)

    def __getitem__(self, idx: int) -> Tuple:
        filename = self._data_filenames[idx]
        filename = os.path.splitext(filename)[0] + ".npy"
        label = self._label_to_idx[filename.split('/')[-2]]

        inputs = np.load(filename)

        return inputs, label

    @property
    def num_classes(self):
        return len(self._label_to_idx)

    @property
    def labels(self):
        return list(self._label_to_idx.keys())
