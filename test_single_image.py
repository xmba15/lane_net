#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torch
from config import Config

from models import LaneNet
from data_loader import TuSimpleDataset
from utils import process_one_image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True)
parser.add_argument("--snapshot", required=True)
parser.add_argument("--overlay_ratio", type=float, default=0.7)
parsed_args = parser.parse_args()


def main(args):
    dt_config = Config()
    val_dataset = TuSimpleDataset(
        data_path=dt_config.DATA_PATH, phase="val", transform=None
    )
    colors = val_dataset.colors
    img = cv2.imread(args.image_path)
    assert img is not None

    model = LaneNet(num_classes=2, embedding_dim=6, img_size=(720, 1280))
    model.load_state_dict(torch.load(args.snapshot)["state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    overlay = process_one_image(model, img, colors, alpha=args.overlay_ratio)
    cv2.imwrite("overlay.png", overlay)


if __name__ == "__main__":
    main(args=parsed_args)
