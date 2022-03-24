from tkinter import W
from numpy import ndarray
import torch
import h5py
from pathlib import Path
from typing import Tuple


def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label


class LAHeart(torch.utils.data.Dataset):

    def __init__(self, split='train', transform=None):
        self._transform = transform

        p = Path(Path(__file__).resolve().parent, 'problem1_datas', 'train')
        self._image_label = [read_h5(f) for f in p.glob('*.h5')]

    def __len__(self):
        # len_sum = 0
        # for image, label in self.image_label:
        # len_sum += len(image)
        return len(self._image_label)

    def __getitem__(self, index):
        # run_idx = index
        # for image, label in self.image_label:
        #     if run_idx < len(image):
        #         return image(run_idx), label(run_idx)
        #     run_idx -= len(image)
        # raise IndexError

        #data: Tuple[ndarray, ndarray] = self._image_label[index]
        image, label = self._image_label[index]
        data = {'image': image, 'label': label}

        # comment out to swap axis from w, h, d to h, w,d
        # data = data[0].swapaxes(0, 1), data[1].swapaxes(0, 1)
        if self._transform:
            data = self._transform(data)
        return data
