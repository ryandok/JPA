import os
import torch
import random
import nibabel as nib
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler


class BC(Dataset):
    def __init__(self, case_list, data_root_path, img_prefix_list=None,
                 label_prefix='TumorMask', transform=None):
        self.case_list = case_list
        self.data_root_path = data_root_path
        self.img_prefix_list = img_prefix_list
        self.label_prefix = label_prefix
        self.transform = transform

    def __getitem__(self, index):
        volume_path_list = [os.path.join(self.data_root_path, img_prefix,
                                         img_prefix + '_' + self.case_list[index]) for img_prefix in
                            self.img_prefix_list]
        label_path = os.path.join(self.data_root_path, self.label_prefix,
                                  self.label_prefix + '_' + self.case_list[index])

        volume_list = [nib.load(volume_path).get_fdata() for volume_path in volume_path_list]
        label = nib.load(label_path).get_fdata()

        name = self.case_list[index].split('.')[0]
        sample = {'name': name, 'label': label}
        for index, volume in enumerate(volume_list):
            sample['volume%d' % (index + 1)] = volume
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.case_list)


class Crop2patchesToTensor(object):
    def __init__(self, output_size, volume_key_list=None,
                 label_key='label'):
        self.output_size = output_size
        self.volume_key_list = volume_key_list
        self.label_key = label_key

    def __call__(self, sample):
        volume_list = [sample[index] for index in self.volume_key_list]
        label = sample[self.label_key]

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            for index in range(len(volume_list)):
                volume_list[index] = np.pad(volume_list[index], [(pw, pw), (ph, ph), (pd, pd)], mode='constant',
                                            constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        # one patch for random
        (w, h, d) = volume_list[0].shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label_random = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        volume_random_list = []
        for index in range(len(volume_list)):
            volume_random_list.append(volume_list[index][w1:w1 + self.output_size[0], h1:h1 + self.output_size[1],
                                      d1:d1 + self.output_size[2]])

        # one patch for foreground
        bbox = self.find_bbox_v2(label)
        if bbox['shape'][0] < self.output_size[0]:
            if max(0, bbox['x2'] - self.output_size[0]) >= min(w - self.output_size[0], bbox['x1']):
                w2 = min(w - self.output_size[0], bbox['x1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                w2 = np.random.randint(max(0, bbox['x2'] - self.output_size[0]),
                                       min(w - self.output_size[0], bbox['x1']))
        else:
            w2 = np.random.randint(max(0, bbox['x1'] - self.output_size[0]), min(w - self.output_size[0], bbox['x2']))

        if bbox['shape'][1] < self.output_size[1]:
            if max(0, bbox['y2'] - self.output_size[1]) >= min(h - self.output_size[1], bbox['y1']):
                h2 = min(h - self.output_size[1], bbox['y1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                h2 = np.random.randint(max(0, bbox['y2'] - self.output_size[1]),
                                       min(h - self.output_size[1], bbox['y1']))
        else:
            h2 = np.random.randint(max(0, bbox['y1'] - self.output_size[1]), min(h - self.output_size[1], bbox['y2']))

        if bbox['shape'][2] < self.output_size[2]:
            if max(0, bbox['z2'] - self.output_size[2]) >= min(d - self.output_size[2], bbox['z1']):
                d2 = min(d - self.output_size[2], bbox['z1'])
                # raise ValueError("low>=up, case name: %s, bbox shape=" % sample['name'] + str(bbox['shape']), bbox, volume.shape)
            else:
                d2 = np.random.randint(max(0, bbox['z2'] - self.output_size[2]),
                                       min(d - self.output_size[2], bbox['z1']))
        else:
            d2 = np.random.randint(max(0, bbox['z1'] - self.output_size[2]), min(d - self.output_size[2], bbox['z2']))

        label_foreground = label[w2:w2 + self.output_size[0], h2:h2 + self.output_size[1], d2:d2 + self.output_size[2]]
        volume_foreground_list = []
        for index in range(len(volume_list)):
            volume_foreground_list.append(volume_list[index][w2:w2 + self.output_size[0], h2:h2 + self.output_size[1],
                                          d2:d2 + self.output_size[2]])

        # concatenate
        for index in range(len(volume_list)):
            volume_list[index] = np.concatenate(
                (np.expand_dims(volume_random_list[index], 0), np.expand_dims(volume_foreground_list[index], 0)))
        label = np.concatenate((np.expand_dims(label_random, 0), np.expand_dims(label_foreground, 0)))

        # To Tensor
        for index in range(len(volume_list)):
            volume_list[index] = torch.Tensor(np.expand_dims(volume_list[index], axis=1).copy())
        label = torch.Tensor(np.expand_dims(label, axis=1).copy())

        for index, key in enumerate(self.volume_key_list):
            sample[key] = volume_list[index]
        sample[self.label_key] = label
        return sample

    def find_bbox(self, array):
        shape = array.shape

        x1 = 0
        x2 = shape[0] - 1
        y1 = 0
        y2 = shape[1] - 1
        z1 = 0
        z2 = shape[2] - 1

        if not (array == 0).all():
            while x1 <= x2 and (array[x1, :, :] == 0).all():
                x1 += 1
            while x1 <= x2 and (array[x2, :, :] == 0).all():
                x2 -= 1

            while y1 <= y2 and (array[:, y1, :] == 0).all():
                y1 += 1
            while y1 <= y2 and (array[:, y2, :] == 0).all():
                y2 -= 1

            while z1 <= z2 and (array[:, :, z1] == 0).all():
                z1 += 1
            while z1 <= z2 and (array[:, :, z2] == 0).all():
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

    def find_bbox_v2(self, array):
        shape = array.shape

        x1 = 0
        x2 = shape[0] - 1
        y1 = 0
        y2 = shape[1] - 1
        z1 = 0
        z2 = shape[2] - 1

        index_list = np.argwhere(array == 1)
        x1 = index_list[:, 0].min()
        x2 = index_list[:, 0].max()
        y1 = index_list[:, 1].min()
        y2 = index_list[:, 1].max()
        z1 = index_list[:, 2].min()
        z2 = index_list[:, 2].max()

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