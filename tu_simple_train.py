#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import time
import os
import math
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from config import Config
from models import LaneNet, LaneNetLoss
from trainer import Trainer
from data_loader import TuSimpleDataset, TuSimpleDataTransform
from trainer import Trainer
from torchsummary import summary
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=4)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr_rate", type=float, default=5e-4)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight_decay", default=2e-4, type=float)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--input_size", default="720,1280", type=str)
parser.add_argument("--save_period", type=int, default=5)
parser.add_argument("--snapshot", type=str)
parsed_args = parser.parse_args()


def main(args):
    dt_config = Config()
    input_size = [int(v.strip()) for v in parsed_args.input_size.split(",")]
    num_classes = 2

    data_transform = TuSimpleDataTransform(
        num_classes=num_classes, input_size=input_size
    )

    train_dataset = TuSimpleDataset(
        data_path=dt_config.DATA_PATH, phase="train", transform=data_transform
    )
    val_dataset = TuSimpleDataset(
        data_path=dt_config.DATA_PATH, phase="val", transform=data_transform
    )
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}

    model = LaneNet(
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        img_size=input_size,
    )

    # run train_dataset.weighted_class()) to calculate the weighted values for
    # each class again
    weighted_values = [1.46884111, 15.9926377]
    criterion = LaneNetLoss(weighted_values=weighted_values)

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=100, gamma=0.1
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        save_period=args.save_period,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
    )

    if parsed_args.snapshot and os.path.isfile(parsed_args.snapshot):
        trainer.resume_checkpoint(parsed_args.snapshot)

    logs = trainer.train()
    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(dt_config.SAVED_MODEL_PATH, "logs.csv"))


if __name__ == "__main__":
    main(parsed_args)
