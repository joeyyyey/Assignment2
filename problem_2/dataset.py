"""Dataset.

    Customize your dataset here.
"""

import os

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path  # added
import csv

from torchvision import transforms
from typing import Tuple


def read_csv(path):
    with open(path, newline='') as csvfile:
        data = csv.reader(csvfile, quotechar='|')
        next(data)
        return [(row[0], int(row[1])) for row in data]
        # return data


class Skin7(Dataset):
    """SKin Lesion"""

    def __init__(self, root="./datas", train=True, transform=None, target_transform=None): # print(f'target is {target}\n')
        #self._root = os.path.join(root)
        self._root = Path(__file__).resolve().parent
        self._transform = transform
        # self._target_transform = target_transform
        self._train = train

        p = Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'train.csv' if train else 'test.csv')

        print(f'Path(__file__).resolve() = {Path(__file__).resolve()}')
        print(f'Path(__file__).resolve().parent = {Path(__file__).resolve().parent}')
        self._imfile_label = read_csv(p)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        imfile, label = self._imfile_label[index]
        image = Image.open(Path(self._root,'datas', 'images', imfile))
        image, label = image, label
        # data = image
        if self._transform:
            image = self._transform(image)
        # if self._target_transform:
        #     label = self._target_transform(label)
        return image, label
        # pass

    def __len__(self):
        return len(self._imfile_label)
