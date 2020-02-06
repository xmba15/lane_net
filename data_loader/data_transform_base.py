#!/usr/bin/env python
# -*- coding: utf-8 -*-
from albumentations import Compose


class DataTransformBase(object):
    def __init__(self):
        self._train_transform_list = []
        self._val_transform_list = []

        self._transforms_dict = {}

    def _initialize_transform_dict(self):
        self._transforms_dict["train"] = Compose(self._train_transform_list)
        self._transforms_dict["val"] = Compose(self._val_transform_list)

    def __call__(self, image, mask=None, phase=None):
        if phase is None:
            return self._transforms_dict["val"](image=image)["image"]

        assert phase in ("train", "val")
        assert mask is not None

        augmented = self._transforms_dict[phase](image=image, masks=[mask])

        return augmented["image"], augmented["masks"][0]
