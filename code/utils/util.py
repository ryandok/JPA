import nibabel as nib
import xlwt
import xlrd
import os
import numpy as np
import torch
import random
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


def onehot(tensor, label_list, device="cuda:0"):
    """
    one hot encoder
    :param tensor:
    :param label_list:
    :param device: cuda:?
    :return:
    """
    tensor = tensor.float()
    shape = list(tensor.shape)
    shape[1] = len(label_list)
    result = torch.zeros(shape).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(tensor.shape), fill_value=label_class).to(device)
        label_seg = (label_mask == tensor).float()
        result[:, index, :, :, :] = label_seg.squeeze(dim=1)
    return result


def standardized_seg(seg, label_list, device="cuda:0"):
    """
    standardized seg_tensor with label list to generate a tensor,
    which can be put into nn.CrossEntropy(input, target) as "target"
    :param seg:
    :param label_list: (include 0)
    :param device: cuda device
    :return:
    """
    seg = torch.squeeze(seg, dim=1)
    result = torch.zeros(seg.shape, dtype=torch.long).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(seg.shape), fill_value=label_class).to(device)
        label_seg = (label_mask == seg).long()
        label_seg = label_seg * index
        result = torch.add(result, label_seg)
    return result


def get_train_val_test_list(fold_root_path, fold_index=0):
    fold_txt_path = os.path.join(fold_root_path, 'fold_%d.txt' % fold_index)
    f = open(fold_txt_path, encoding='gbk')
    txt = []
    for line in f:
        txt.append(line.strip())

    train_index = txt.index('train:')
    val_index = txt.index('val:')
    test_index = txt.index('test:')

    train_list = txt[train_index + 1:val_index]
    val_list = txt[val_index + 1:test_index]
    test_list = txt[test_index + 1:]

    return train_list, val_list, test_list


def get_test_list(txt_path):
    f = open(txt_path, encoding='gbk')
    test_list = []
    for line in f:
        test_list.append(line.strip())

    return test_list


# if __name__ == '__main__':
#     a = get_test_list('../../data/test_495.txt')
#     print(1)





