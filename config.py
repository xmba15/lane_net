#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    def __init__(self):
        self.CURRENT_DIR = _CURRENT_DIR

        self.DATA_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "data"))

        self.SAVED_MODEL_PATH = os.path.abspath(
            os.path.join(_CURRENT_DIR, "saved_models")
        )
        if not os.path.isdir(self.SAVED_MODEL_PATH):
            os.system("mkdir -p {}".format(self.SAVED_MODEL_PATH))

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
