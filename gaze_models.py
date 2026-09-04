# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Claudia Resendiz-Jurado and contributors

"""Architecture required to load the gaze-estimation checkpoint."""

import torch.nn as nn
import torchvision.models as models


def _extract_sd(data):
    """Extract a state dictionary from any supported checkpoint container."""
    if isinstance(data, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in data:
                return data[key]
    return data


def get_gaze_model():
    """Build the ResNet-101 architecture used to train resnet101_v2."""
    backbone = models.resnet101(weights=None)
    features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(features, 512),
        nn.LayerNorm(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.30),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.15),
        nn.Linear(256, 2),
    )
    return backbone
