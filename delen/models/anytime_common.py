#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AnytimeSubNetwork(nn.Module):
    """ An anytime sub-network consisted of two parts:
            1. Feature extractor
            2. Classifier / Predictor
        This module should return the output of both parts
    """
    def __init__(self,
                 feature_extractor: nn.Module,
                 classifier: nn.Module,
                 include_features=False):
        super(AnytimeSubNetwork, self).__init__()
        self.include_features = include_features
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, inputs: torch.Tensor):
        features = self.feature_extractor(inputs)
        output = self.classifier(features)

        if self.include_features:
            return output, features
        else:
            return output