#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torch
import tqdm
from config import Config

from models import LaneNet
from data_loader import TuSimpleDataset
from utils import process_one_image

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--snapshot",
    default=os.path.join(
        _CURRENT_DIR, "saved_models/checkpoint_LaneNet_epoch_99.pth"
    ),
)
parser.add_argument("--embedding_dim", type=int, default=4)
parser.add_argument("--input_size", default="720,1280", type=str)
parser.add_argument("--overlay_ratio", type=float, default=0.7)
parsed_args = parser.parse_args()


def human_sort(s):
    """Sort list the way humans do
    """
    import re

    pattern = r"([0-9]+)"
    return [int(c) if c.isdigit() else c.lower() for c in re.split(pattern, s)]


def main(args):
    dt_config = Config()
    input_size = [int(v.strip()) for v in parsed_args.input_size.split(",")]
    num_classes = 2

    val_dataset = TuSimpleDataset(
        data_path=dt_config.DATA_PATH, phase="val", transform=None
    )
    colors = val_dataset.colors
    model = LaneNet(
        num_classes=num_classes,
        embedding_dim=parsed_args.embedding_dim,
        img_size=input_size,
    )
    model.load_state_dict(torch.load(args.snapshot)["state_dict"])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    image_path = os.path.join(_CURRENT_DIR, "test_images")
    image_names = os.listdir(image_path)
    image_names = sorted(image_names, key=human_sort)
    image_names = [
        os.path.join(image_path, image_name) for image_name in image_names
    ]

    for image_name in tqdm.tqdm(image_names):
        img = cv2.imread(image_name)
        overlay = process_one_image(
            model, img, colors, img_size=input_size, alpha=args.overlay_ratio
        )
        cv2.imshow("overlay", cv2.resize(overlay, (720, 360)))
        # cv2.imwrite("overlay.png", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(args=parsed_args)
