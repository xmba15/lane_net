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
    ShiftScaleRotate,
    RandomResizedCrop

)
from albumentations.pytorch import ToTensor
import random


class TuSimpleDataTransform(DataTransformBase):
    def __init__(self, num_classes, input_size):
        super(TuSimpleDataTransform, self).__init__()

        random.seed(1000)
        height, width = input_size

        self._train_transform_list = self._train_transform_list + [
            HorizontalFlip(p=0.5),
            GaussNoise(p=0.5),
            RandomBrightnessContrast(p=0.5),
            RandomShadow(p=0.5),
            RandomRain(rain_type="drizzle", p=0.5),
            ShiftScaleRotate(rotate_limit=10, p=0.5),
            RandomResizedCrop(height=height, width=width, scale=(0.8, 1), p=0.5),
        ]

        self._train_transform_list.append(Resize(height, width))
        self._val_transform_list.append(Resize(height, width))

        self._train_transform_list.append(ToTensor(num_classes=num_classes))
        self._val_transform_list.append(ToTensor(num_classes=num_classes))

        self._initialize_transform_dict()
