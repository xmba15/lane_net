#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
from .trainer_base import TrainerBase


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import inf_loop
except:
    print("cannot load modules")
    sys.exit(-1)


class Trainer(TrainerBase):
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        data_loaders_dict,
        scheduler=None,
        device=None,
        len_epoch=None,
        dataset_name_base="",
        alpha=1.0,
    ):
        super(Trainer, self).__init__(
            model,
            criterion,
            metric_func,
            optimizer,
            num_epochs,
            save_period,
            config,
            device,
            dataset_name_base,
        )

        self.train_data_loader = data_loaders_dict["train"]
        self.val_data_loader = data_loaders_dict["val"]
        if len_epoch is None:
            self._len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(self.train_data_loader)
            self._len_epoch = len_epoch

        self._do_validation = self.val_data_loader is not None
        self._scheduler = scheduler
        self._alpha = alpha

    def _train_epoch(self, epoch):
        self._model.train()

        epoch_train_loss = 0.0
        for batch_idx, (data, target) in enumerate(self.train_data_loader):
            data, target = data.to(self._device), target.to(self._device)
            target = target.long()

            self._optimizer.zero_grad()

            output = self._model(data)
            seg_loss, var_loss, dist_loss = self._criterion(output, target)
            train_loss = seg_loss + var_loss + dist_loss
            train_loss.backward()
            self._optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    "\n epoch: {} || iter: {} || seg_loss: {} || var_loss: {} || dist_loss:{} || total_loss: {}".format(
                        epoch,
                        batch_idx,
                        seg_loss.item(),
                        var_loss.item(),
                        dist_loss.item(),
                        train_loss.item(),
                    )
                )

            epoch_train_loss += train_loss.item()
            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            epoch_val_loss = self._valid_epoch(epoch)

        if self._scheduler is not None:
            self._scheduler.step()

        epoch_train_loss /= len(self.train_data_loader)

        return epoch_train_loss, epoch_val_loss

    def _valid_epoch(self, epoch):
        print("start validation...")
        self._model.eval()

        epoch_val_loss = 0.0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_data_loader):
                data, target = data.to(self._device), target.to(self._device)
                target = target.long()

                output = self._model(data)
                seg_loss, var_loss, dist_loss = self._criterion(output, target)
                val_loss = seg_loss + var_loss + dist_loss
                epoch_val_loss += val_loss.item()

        return epoch_val_loss / len(self.val_data_loader)
