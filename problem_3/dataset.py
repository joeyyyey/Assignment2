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
        list_tuple = [(row[0], int(row[1])) for row in data]
        return list_tuple #{(row[0], int(row[1])) for row in data}
        # return data


class Cholec80(Dataset):
    """Surgical Dataset"""

    def __init__(self, root="./datas", train=True, transform=None): # print(f'target is {target}\n')
        # self._root = os.path.join(root)
        self._root = Path(__file__).resolve().parent
        self._transform = transform
        # self._target_transform = target_transform
        self._train = train
        
        if train:
            p_list = [Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'video_1.csv'), Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'video_2.csv'),Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'video_3.csv'),Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'video_4.csv'),Path(
            Path(__file__).resolve().parent, 'datas', 'annotation',
            'video_5.csv')]
            p_img_list = [Path(
            Path(__file__).resolve().parent, 'datas', '1'), Path(
            Path(__file__).resolve().parent, 'datas', '2'), Path(
            Path(__file__).resolve().parent, 'datas', '3'), Path(
            Path(__file__).resolve().parent, 'datas', '4'), Path(
            Path(__file__).resolve().parent, 'datas', '5')]
        else:
            p_list = [Path(
                Path(__file__).resolve().parent, 'datas', 'annotation',
                'video_41.csv')]
            p_img_list = [Path(
                Path(__file__).resolve().parent, 'datas', '41')]
        # for p in p_list:
            # for 
            # self._imfile_label = read_csv(p) # train: list of 5 lists of tuples (2)
        self._p_img_list = p_img_list
        self._imfile_label = [(img, label) for f in p_list for (img, label) in read_csv(f)]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the
                   target class.
        """
        # for i in range(len(self._imfile_label)):
        #     for imfile, label in self._imfile_label:
        #         imfile, label = self._imfile_label[index][i]
                    
        #         image = Image.open(Path(self._root, imfile))
        #         image, label = image, label
        #         if self._transform:
        #             image = self._transform(image)
        imfile, label = self._imfile_label[index]
        # image = Image.open(Path(self._root, imfile))
        for p in self._p_img_list:
            path = p
            image = Image.open(Path(p, imfile))

        image, label = image, label
        if self._transform:
            image = self._transform(image)
        return image, label
        # pass

    def __len__(self):
        return len(self._imfile_label) #3575 fot train, 777 for test
