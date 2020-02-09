#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import json
import tqdm
import torch.utils.data as data


class TuSimpleDataset(data.Dataset):
    def __init__(
        self,
        data_path,
        classes=["bg", "lane"],
        phase="train",
        transform=None,
        max_instances=100,
        seg_width=30,
    ):
        super(TuSimpleDataset, self).__init__()

        assert os.path.isdir(data_path)
        assert phase in ("train", "val", "test")

        self._data_path = os.path.join(data_path, "tu_simple")
        self._classes = classes
        self._transform = transform
        self._phase = phase
        self._max_instances = max_instances
        self.colors = TuSimpleDataset.generate_color_chart(self._max_instances)
        self._seg_width = seg_width

        self._path_dicts = {
            "train": "train_set",
            "val": "train_set",
            "test": "test_set",
        }

        self._json_dicts = {
            "train": [
                "train_set/label_data_0313.json",
                "train_set/label_data_0601.json",
            ],
            "val": ["train_set/label_data_0531.json"],
            "test": ["test_set/test_label.json"],
        }
        self._load_ground_truths()

    def __len__(self):
        return len(self.image_paths)

    def _load_ground_truths(self):
        self.image_paths = []
        self.targets = []
        self.targets = []

        image_path_base = os.path.join(
            self._data_path, self._path_dicts[self._phase]
        )
        json_files = [
            os.path.join(self._data_path, json_path)
            for json_path in self._json_dicts[self._phase]
        ]
        for json_file in json_files:
            with open(json_file) as f:
                for line in f:
                    label = json.loads(line)

                    all_lanes = []
                    one_lane = []
                    slopes = (
                        []
                    )  # identify 1st, 2nd, 3rd, 4th lane through slope
                    for i in range(len(label["lanes"])):
                        l = [
                            (x, y)
                            for x, y in zip(
                                label["lanes"][i], label["h_samples"]
                            )
                            if x >= 0
                        ]
                        if len(l) > 1:
                            one_lane.append(l)
                            slopes.append(
                                np.arctan2(
                                    l[-1][1] - l[0][1], l[0][0] - l[-1][0]
                                )
                                / np.pi
                                * 180
                            )
                            one_lane = [one_lane[i] for i in np.argsort(slopes)]
                            slopes = [slopes[i] for i in np.argsort(slopes)]

                    idx_1 = None
                    idx_2 = None
                    idx_3 = None
                    idx_4 = None
                    for i, slope in enumerate(slopes):
                        if slope <= 90:
                            idx_2 = i
                            idx_1 = i - 1 if i > 0 else None
                        else:
                            idx_3 = i
                            idx_4 = i + 1 if i + 1 < len(slopes) else None
                            break
                    all_lanes.append([] if idx_1 is None else one_lane[idx_1])
                    all_lanes.append([] if idx_2 is None else one_lane[idx_2])
                    all_lanes.append([] if idx_3 is None else one_lane[idx_3])
                    all_lanes.append([] if idx_4 is None else one_lane[idx_4])

                    abs_img_path = os.path.join(
                        image_path_base, label["raw_file"]
                    )
                    self.image_paths.append(abs_img_path)
                    self.targets.append(all_lanes)

    def __getitem__(self, idx):
        image, gt_label = self._pull_item(idx)

        return image, gt_label

    def _pull_item(self, idx):
        assert idx < self.__len__()
        abs_img_path = self.image_paths[idx]
        img = cv2.imread(abs_img_path)
        target = self.targets[idx]
        height, width, _ = img.shape
        gt_label = np.zeros((height, width, 3))
        for i, coords in enumerate(target):
            if len(coords) < 4:
                continue
            for j in range(len(coords) - 1):
                cv2.line(
                    gt_label,
                    coords[j],
                    coords[j + 1],
                    (i + 1, i + 1, i + 1),
                    self._seg_width // 2,
                )

        gt_label = gt_label[:, :, 0].astype("int64")

        if self._transform is not None:
            img, gt_label = self._transform(img, gt_label, self._phase)

        return img, gt_label

    def get_overlay_image(self, idx=None, image=None, label=None, alpha=0.5):
        if image is None or label is None:
            assert idx is not None and idx < self.__len__()
            image, label = self.__getitem__(idx)

        mask = np.array(self.colors)[label]
        overlay = (((1 - alpha) * image) + (alpha * mask)).astype("uint8")

        return overlay

    def weighted_class(self):
        assert self.__len__() > 0
        print("Estimating weights for each class")

        class_dist_dict = dict((el, 0) for el in self._classes)
        class_idx_dict = TuSimpleDataset.class_to_class_idx_dict(self._classes)

        for idx in tqdm.tqdm(range(self.__len__())):
            _, gt = self.__getitem__(idx)
            gt[gt > 0] = 1
            for class_name in self._classes:
                class_dist_dict[class_name] += np.count_nonzero(
                    gt == class_idx_dict[class_name]
                )

        total_pixels = np.sum(list(class_dist_dict.values()))

        weighted = np.zeros(len(self._classes), dtype=np.float64)
        for key, value in class_dist_dict.items():
            weighted[class_idx_dict[key]] = 1 / np.log(
                value * 1.0 / total_pixels + 1.02
            )

        return weighted

    @staticmethod
    def class_to_class_idx_dict(classes):
        class_idx_dict = {}

        for i, class_name in enumerate(classes):
            class_idx_dict[class_name] = i

        return class_idx_dict

    @staticmethod
    def generate_color_chart(num_classes, seed=1812):
        assert num_classes > 0
        np.random.seed(seed)

        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype="uint8")
        colors = np.vstack([colors]).astype("uint8")
        colors = [tuple(color) for color in list(colors)]
        colors = [tuple(int(e) for e in color) for color in colors]

        return colors
