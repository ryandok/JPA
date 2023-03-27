import os
import torch
import random
import nibabel as nib
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import itertools


class BC(Dataset):
    def __init__(self, case_list, data_root_path, img_prefix='DCE-C1', label_prefix='TumorBreast', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.img_prefix = img_prefix
        self.label_prefix = label_prefix
        self.transform = transform

    def __getitem__(self, index):
        volume_path = os.path.join(self.data_root_path, self.img_prefix,
                                   self.img_prefix + '_' + self.case_list[index])
        label_path = os.path.join(self.data_root_path, self.label_prefix,
                                  self.label_prefix + '_' + self.case_list[index])

        volume = nib.load(volume_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        name = self.case_list[index].split('.')[0]
        sample = {'name': name, 'volume': volume, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.case_list)


class CenterCrop(object):
    def __init__(self, output_size, volume_key='volume', label_key='label'):
        self.output_size = output_size
        self.volume_key = volume_key
        self.label_key = label_key

    def __call__(self, sample):
        volume, label = sample[self.volume_key], sample[self.label_key]

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume = np.pad(volume, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = volume.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume = volume[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        sample[self.volume_key], sample[self.label_key] = volume, label
        return sample


class RandomCrop(object):
    def __init__(self, output_size, volume_key='volume', label_key='label'):
        self.output_size = output_size
        self.volume_key = volume_key
        self.label_key = label_key

    def __call__(self, sample):
        volume, label = sample[self.volume_key], sample[self.label_key]

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume = np.pad(volume, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = volume.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume = volume[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        sample[self.volume_key], sample[self.label_key] = volume, label
        return sample


class Crop2patchesToTensor(object):
    def __init__(self, output_size, volume_key='volume', label_key='label'):
        self.output_size = output_size
        self.volume_key = volume_key
        self.label_key = label_key

    def __call__(self, sample):
        volume, label = sample[self.volume_key], sample[self.label_key]

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            volume = np.pad(volume, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # one patch for random
        (w, h, d) = volume.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label_random = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume_random = volume[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        # one patch for foreground
        bbox = self.find_bbox(label)
        if bbox['shape'][0] < self.output_size[0]:
            if max(0, bbox['x2'] - self.output_size[0]) >= min(w - self.output_size[0], bbox['x1']):
                w2 = min(w - self.output_size[0], bbox['x1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                w2 = np.random.randint(max(0, bbox['x2'] - self.output_size[0]), min(w - self.output_size[0], bbox['x1']))
        else:
            w2 = np.random.randint(max(0, bbox['x1'] - self.output_size[0]), min(w - self.output_size[0], bbox['x2']))

        if bbox['shape'][1] < self.output_size[1]:
            if max(0, bbox['y2'] - self.output_size[1]) >= min(h - self.output_size[1], bbox['y1']):
                h2 = min(h - self.output_size[1], bbox['y1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                h2 = np.random.randint(max(0, bbox['y2'] - self.output_size[1]), min(h - self.output_size[1], bbox['y1']))
        else:
            h2 = np.random.randint(max(0, bbox['y1'] - self.output_size[1]), min(h - self.output_size[1], bbox['y2']))

        if bbox['shape'][2] < self.output_size[2]:
            if max(0, bbox['z2'] - self.output_size[2]) >= min(d - self.output_size[2], bbox['z1']):
                d2 = min(d - self.output_size[2], bbox['z1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                d2 = np.random.randint(max(0, bbox['z2'] - self.output_size[2]), min(d - self.output_size[2], bbox['z1']))
        else:
            d2 = np.random.randint(max(0, bbox['z1'] - self.output_size[2]), min(d - self.output_size[2], bbox['z2']))

        label_foreground = label[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        volume_foreground = volume[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]

        # concatenate
        volume = np.concatenate((np.expand_dims(volume_random, 0), np.expand_dims(volume_foreground, 0)))
        label = np.concatenate((np.expand_dims(label_random, 0), np.expand_dims(label_foreground, 0)))

        # To Tensor
        volume = torch.Tensor(np.expand_dims(volume, axis=1).copy())
        label = torch.Tensor(np.expand_dims(label, axis=1).copy())

        sample[self.volume_key], sample[self.label_key] = volume, label
        return sample

    def find_bbox(self, array):
        shape = array.shape

        x1 = 0
        x2 = shape[0] - 1
        y1 = 0
        y2 = shape[1] - 1
        z1 = 0
        z2 = shape[2] - 1

        if not (array==0).all():
            while (array[x1, :, :] == 0).all():
                x1 += 1
            while (array[x2, :, :] == 0).all():
                x2 -= 1

            while (array[:, y1, :] == 0).all():
                y1 += 1
            while (array[:, y2, :] == 0).all():
                y2 -= 1

            while (array[:, :, z1] == 0).all():
                z1 += 1
            while (array[:, :, z2] == 0).all():
                z2 -= 1

        x_len = x2 - x1 + 1
        y_len = y2 - y1 + 1
        z_len = z2 - z1 + 1
        shape = (x_len, y_len, z_len)
        bbox = {'x1': x1,
                'x2': x2,
                'y1': y1,
                'y2': y2,
                'z1': z1,
                'z2': z2,
                'shape': shape}
        return bbox


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        secondary_iter = iterate_once(self.secondary_indices)
        primary_iter = iterate_eternally(self.primary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.secondary_indices) // self.secondary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    " Collect data into fixed-length chunks or blocks "
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



