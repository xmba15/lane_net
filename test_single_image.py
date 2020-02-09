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
parser.add_argument("--embedding_dim", type=int, default=4)
parser.add_argument("--input_size", default="720,1280", type=str)
parser.add_argument("--overlay_ratio", type=float, default=0.7)
parsed_args = parser.parse_args()


def main(args):
    dt_config = Config()
    input_size = [int(v.strip()) for v in parsed_args.input_size.split(",")]
    num_classes = 2

    val_dataset = TuSimpleDataset(
        data_path=dt_config.DATA_PATH, phase="val", transform=None
    )
    colors = val_dataset.colors
    img = cv2.imread(args.image_path)
    assert img is not None

    model = LaneNet(
        num_classes=num_classes,
        embedding_dim=parsed_args.embedding_dim,
        img_size=input_size,
    )
    model.load_state_dict(torch.load(args.snapshot)["state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    overlay = process_one_image(model, img, colors, img_size=input_size, alpha=args.overlay_ratio)
    cv2.imwrite("overlay.png", overlay)


if __name__ == "__main__":
    main(args=parsed_args)
