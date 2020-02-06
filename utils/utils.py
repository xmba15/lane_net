#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth


def inf_loop(data_loader):
    from itertools import repeat

    for loader in repeat(data_loader):
        yield from loader


def embedding_post_process(embedding, bin_seg, band_width=1.5, max_num_lane=4):
    """
    First use mean shift to find dense cluster center.
    Arguments:
    ----------
    embedding: numpy [H, W, embed_dim]
    bin_seg: numpy [H, W], each pixel is 0 or 1, 0 for background pixel
    delta_v: coordinates within distance of 2*delta_v to cluster center are
    Return:
    ---------
    cluster_result: numpy [H, W], index of different lanes on each pixel
    """
    cluster_result = np.zeros(bin_seg.shape, dtype=np.int32)

    cluster_list = embedding[bin_seg > 0]
    if len(cluster_list) == 0:
        return cluster_result

    mean_shift = MeanShift(bandwidth=1.5, bin_seeding=True)
    mean_shift.fit(cluster_list)

    labels = mean_shift.labels_
    cluster_result[bin_seg > 0] = labels + 1

    cluster_result[cluster_result > max_num_lane] = 0
    for idx in np.unique(cluster_result):
        if len(cluster_result[cluster_result == idx]) < 15:
            cluster_result[cluster_result == idx] = 0

    return cluster_result


def process_one_image(model, img, colors, alpha=0.3):
    import torch
    import numpy as np

    processed_img = img / 255.0
    # [np.newaxis,:] equals unsqueeze(0)
    processed_img = torch.tensor(
        processed_img.transpose(2, 0, 1)[np.newaxis, :]
    ).float()
    if torch.cuda.is_available():
        processed_img = processed_img.cuda()

    embedding_output, segmentation_output = model(processed_img)
    embedding_output = embedding_output.detach().cpu().numpy()
    embedding_output = np.transpose(embedding_output[0], (1, 2, 0))

    mask = (
        segmentation_output.data.max(1)[1]
        .cpu()
        .numpy()
        .reshape(img.shape[0], img.shape[1])
    )

    mask = embedding_post_process(embedding_output, mask)

    color_mask = np.array(colors)[mask]
    overlay = np.copy(img)
    overlay[mask != 0] = (
        (1 - alpha) * overlay[mask != 0] + alpha * color_mask[mask != 0]
    ).astype("uint8")

    return overlay
