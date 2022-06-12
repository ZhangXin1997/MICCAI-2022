import numpy as np
import random

import torch

import skimage.io
import skimage.transform
import skimage.color
import skimage


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h, w, c = img.shape

        if random.random() < self.p:
            img = img[:, ::-1, :]
            if label != []:
                label[:, [0, 2]] = w - label[:, [2, 0]]

            img = np.ascontiguousarray(img)
            sample = {'img': img, 'label': label}

        return sample


class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        h, w, c = img.shape

        if random.random() < self.p:
            img = img[::-1, :, :]
            if label != []:
                label[:, [1, 3]] = h - label[:, [3, 1]]

            img = np.ascontiguousarray(img)
            sample = {'img': img, 'label': label}

        return sample


class RandomRotate90(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        H, W, C = img.shape
        if random.random() < 0.5:
            img = np.rot90(img, 1, (0, 1))
            if label != []:
                x = label[:, 0].copy()
                y = label[:, 1].copy()
                w = (label[:, 2] - label[:, 0]).copy()
                h = (label[:, 3] - label[:, 1]).copy()
                label[:, 0] = (W - H) // 2 + y
                label[:, 1] = (W + H) // 2 - x - w
                label[:, 2] = label[:, 0] + h
                label[:, 3] = label[:, 1] + w

            img = np.ascontiguousarray(img)
            sample = {'img': img, 'label': label}
        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        img, label = sample['img'], sample['label']
        # return {'img': torch.from_numpy(((img - self.mean) / self.std)).float(), 'label': torch.from_numpy(label)}
        return {'img': torch.from_numpy(img).float(), 'label': torch.from_numpy(label)}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
