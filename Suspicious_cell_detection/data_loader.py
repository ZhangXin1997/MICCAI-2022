from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from PIL import Image

import cfg


class CervicalDataset(Dataset):
    """Cervical dataset."""

    def __init__(self, data_path, patch_size, transform=None):
        self.data_path = data_path
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img, label = self.load_data(idx)
        sample = {'img': img, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_data(self, index):
        data = np.load(self.data_path[index])
        img, label = data['img'], data['label']/1.0
        img = img.astype(np.float32) / 255.0
    

        return img, label


class WsiDataset(Dataset):
    def __init__(self, read, y_num, x_num, strides, coordinates, patch_size, transform=None):
        self.read = read
        self.y_num = y_num
        self.x_num = x_num
        self.strieds = strides
        self.coordinates = coordinates
        self.patch_size = patch_size
        self.transform = transform

    def __getitem__(self, index):
        coord_y, coord_x = self.coordinates[index]
        img = self.read.ReadRoi(coord_x, coord_y, cfg.patch_size[0], cfg.patch_size[1], scale=20).copy()

        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255.0

        return torch.from_numpy(img).float(), coord_y, coord_x

    def __len__(self):
        return self.y_num * self.x_num


def collater(data):
    imgs = [s['img'] for s in data]
    labels = [s['label'] for s in data]

    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.permute((0, 3, 1, 2))

    max_num_labels = max(label.shape[0] for label in labels)

    if max_num_labels > 0:
        label_pad = torch.ones((len(labels), max_num_labels, 4)) * -1

        for idx, label in enumerate(labels):
            if label.shape[0] > 0:
                label_pad[idx, :label.shape[0], :] = label
    else:
        label_pad = torch.ones((len(labels), 1, 4)) * -1

    return {'img': imgs, 'label': label_pad}



if __name__ == '__main__':
    sample_path = [os.path.join(cfg.train_sample_path, x) for x in os.listdir(cfg.train_sample_path) if '.npz' in x]
    train_data = CervicalDataset(sample_path, cfg.patch_size)
    # for i in range(train_data.__len__()):
    #     print(i)
    #     train_data.__getitem__(i)
