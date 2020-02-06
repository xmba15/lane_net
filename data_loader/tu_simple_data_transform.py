#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .data_transform_base import DataTransformBase
from albumentations import (
    Resize,
    HorizontalFlip,
    Normalize,
    GaussNoise,
    RandomBrightnessContrast,
    RandomShadow,
    RandomRain,
    Rotate,
)
from albumentations.pytorch import ToTensor
import random


class TuSimpleDataTransform(DataTransformBase):
    def __init__(self, num_classes, input_size=None):
        super(TuSimpleDataTransform, self).__init__()

        random.seed(1000)

        if input_size is not None:
            height, width = input_size
            self._train_transform_list.append(Resize(height, width))
            self._val_transform_list.append(Resize(height, width))

        self._train_transform_list = self._train_transform_list + [
            HorizontalFlip(p=0.5),
            Rotate(p=0.5, limit=(-10, 10)),
            GaussNoise(p=0.5),
            RandomBrightnessContrast(p=0.5),
            ToTensor(num_classes=num_classes),
        ]

        self._val_transform_list = self._val_transform_list + [
            ToTensor(num_classes=num_classes),
        ]

        self._initialize_transform_dict()
