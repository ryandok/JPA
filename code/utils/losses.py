import torch
import torch.nn.functional as F
import numpy as np
import math


def gradient_loss(s, penalty='l2'):
    """
    Transformation smoothness
    :param s: flow(one of an output of VoxelMorph)
    :param penalty: choice of penalty
    :return: gradient_loss
    """
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def ncc_loss(I, J):
    mean_I = I.mean([1, 2, 3, 4], keepdim=True)
    mean_J = J.mean([1, 2, 3, 4], keepdim=True)
    I2 = I * I
    J2 = J * J
    mean_I2 = I2.mean([1, 2, 3, 4], keepdim=True)
    mean_J2 = J2.mean([1, 2, 3, 4], keepdim=True)
    stddev_I = torch.sqrt(mean_I2 - mean_I * mean_I).sum([1, 2, 3, 4], keepdim=True)
    stddev_J = torch.sqrt(mean_J2 - mean_J * mean_J).sum([1, 2, 3, 4], keepdim=True)

    return -torch.mean((I - mean_I) * (J - mean_J) / (stddev_I * stddev_J))


def ncc_loss_v2(I, J):
    mean_I = I.mean([1, 2, 3, 4], keepdim=True)
    mean_J = J.mean([1, 2, 3, 4], keepdim=True)
    I2 = I * I
    J2 = J * J
    mean_I2 = I2.mean([1, 2, 3, 4], keepdim=True)
    mean_J2 = J2.mean([1, 2, 3, 4], keepdim=True)
    stddev_I = torch.sqrt(mean_I2 - mean_I * mean_I).sum([1, 2, 3, 4], keepdim=True)
    stddev_J = torch.sqrt(mean_J2 - mean_J * mean_J).sum([1, 2, 3, 4], keepdim=True)

    return 1-torch.mean((I - mean_I) * (J - mean_J) / (stddev_I * stddev_J))


def ncc_loss_v3(I, J, win=None):
    """
    calculate the normalize cross correlation between I and J
    assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    """

    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    if win is None:
        win = [9] * ndims

    # conv_fn = getattr(F, 'conv%dd' % ndims)
    # I2 = I * I
    # J2 = J * J
    # IJ = I * J

    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    return I_var, J_var, cross


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def bce3d_new(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.2 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction) / num_pos


def noise_robust_dice_loss(input, target, gama=1.5):
    target = target.float()
    smooth = 1e-5
    up = torch.sum(torch.pow(torch.abs(input-target), gama))
    down = torch.sum(torch.pow(input, 2)) + torch.sum(torch.pow(target, 2)) + smooth
    loss = up / down
    return loss


