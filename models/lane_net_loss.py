#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneNetLoss(nn.Module):
    def __init__(
        self, delta_v=0.5, delta_d=3.0, weighted_values=None, size_average=None
    ):
        super(LaneNetLoss, self).__init__()

        self._delta_v = delta_v
        self._delta_d = delta_d

        self._seg_loss_func = nn.CrossEntropyLoss(weighted_values, size_average)

    def _discriminative_loss(self, embedding_predictions, targets):
        device = embedding_predictions.device
        batch_size = embedding_predictions.shape[0]
        embedding_dim = embedding_predictions.shape[1]

        var_loss = torch.tensor(
            0, dtype=embedding_predictions.dtype, device=device
        )
        dist_loss = torch.tensor(
            0, dtype=embedding_predictions.dtype, device=device,
        )

        for idx in range(batch_size):
            # (embed_dim, H, W)
            cur_embedding = embedding_predictions[idx]
            cur_seg_gt = targets[idx]

            labels = torch.unique(cur_seg_gt)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # https://github.com/harryhan618/LaneNet/issues/12
                nonsense = targets.sum()
                zero_values = torch.zeros_like(nonsense)
                var_loss = var_loss + nonsense * zero_values
                dist_loss = dist_loss + nonsense * zero_values
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_idx = cur_seg_gt == lane_idx
                if not seg_mask_idx.any():
                    continue
                embedding_i = cur_embedding[:, seg_mask_idx]
                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)

                var_loss = (
                    var_loss
                    + torch.mean(
                        F.relu(
                            torch.norm(
                                embedding_i - mean_i.reshape(embedding_dim, 1),
                                dim=0,
                            )
                            - self._delta_v
                        )
                        ** 2
                    )
                    / num_lanes
                )

            # (n_lane, embed_dim)
            centroid_mean = torch.stack(centroid_mean)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embedding_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embedding_dim)

                # shape (num_lanes, num_lanes)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)

                # diagonal elements are 0, now mask above delta_d
                dist = dist + torch.eye(
                    num_lanes, dtype=dist.dtype, device=device
                )

                dist_loss = (
                    dist_loss
                    + torch.sum(F.relu(-dist + self._delta_d) ** 2)
                    / (num_lanes * (num_lanes - 1))
                    / 2
                )

        return var_loss, dist_loss

    def forward(self, predictions, targets):
        embedding_predictions, segmentation_predictions = predictions
        batch_size = embedding_predictions.shape[0]

        var_loss, dist_loss = self._discriminative_loss(
            embedding_predictions, targets
        )

        seg_loss = self._seg_loss_func(
            segmentation_predictions, torch.gt(targets, 0).long()
        )

        seg_loss /= batch_size
        var_loss /= batch_size
        dist_loss /= batch_size

        return seg_loss, var_loss, dist_loss
