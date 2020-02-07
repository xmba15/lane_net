#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import download_file_from_google_drive
except Exception as e:
    print("cannot import module")
    exit(0)


def main():
    data_path = os.path.join(_CURRENT_DIR, "../data")
    file_id = "1-VMsdeKxOYATf4xC1qrPr_MtkD3SeVkN"
    destination = os.path.join(data_path, "tu_simple.zip")
    if not os.path.isfile(destination) and not os.path.isdir(
        os.path.join(data_path, "tu_simple")
    ):
        download_file_from_google_drive(file_id, destination)
        os.system(
            "cd {} && unzip tu_simple.zip -d {}".format(data_path, data_path)
        )


if __name__ == "__main__":
    main()
