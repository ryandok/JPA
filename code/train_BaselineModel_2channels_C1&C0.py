"""
train Baseline Model (ResUnet) on (C1, C0)
Exp2:
Methods -- CSegNet w/o JPA
BSegNet -- No
CSegNet Input -- C1&C0
"""

# external imports
import os
import argparse
import json
import sys
import time
import datetime
import codecs
import logging
import random
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from medpy import metric
from skimage.transform import resize
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# internal imports
from utils import util
from utils.sliding_window_inference_Nchannels import test_single_case_Nchannel as test_single_case
from utils.losses import dice_loss
from dataloaders.BC_Nchannels import BC
from dataloaders.BC_Nchannels import Crop2patchesToTensor
from dataloaders.data_augmentation_Nchannels import *
from networks.BaselineModel import ResUnet


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='Baseline_CropZeroDown0.5XY_C1&C0_DatalistSiemens',
                    help='experiment name used for save')
parser.add_argument("--gpu", type=str, default='4', help="gpu id")
parser.add_argument("--seed1", type=int, default='1997', help="seed 1")
parser.add_argument("--max_iterations", type=int, default=60000, help="number of iterations of training")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--n_save_iter", type=int, default=1000, help="save the model every time")
parser.add_argument("--n_val_iter", type=int, default=1000, help="val the model every time")
parser.add_argument("--data_root_path", type=str,
                    default='/raid/hra/dataset/BreastCancerMRI_350/CropZero_down0.5XY/Norm/', help='dataset root path')
parser.add_argument("--img1_prefix", type=str, default='DCE-C1', help="img prefix")
parser.add_argument("--img2_prefix", type=str, default='DCE-C0', help="img prefix")
parser.add_argument("--label_prefix", type=str, default='TumorMask', help="label prefix")
parser.add_argument("--datalist_json_path", type=str, default='../data/datalist_Siemens.json', help="datalist path")
parser.add_argument("--model_dir_root_path", type=str, default='../model/', help="root path to save the model")
parser.add_argument("--note", type=str, default="", help="note")
arg = parser.parse_args()


def train(exp_name,
          gpu,
          seed1,
          max_iterations,
          lr,
          n_save_iter,
          n_val_iter,
          data_root_path,
          img1_prefix,
          img2_prefix,
          label_prefix,
          datalist_json_path,
          model_dir_root_path,
          note):

    """ setting """
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed1)
    np.random.seed(seed1)
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed1)

    # time
    now = time.localtime()
    now_format = time.strftime("%Y-%m-%d %H:%M:%S", now)  # time format
    date_now = now_format.split(' ')[0]
    time_now = now_format.split(' ')[1]

    # save model path
    save_path = os.path.join(model_dir_root_path, exp_name)
    if os.path.exists(save_path):
        print('Experiment name "%s" repeat, please check!!!' % exp_name)
        return
    os.makedirs(save_path)

    # print setting
    print("----------------------------------setting-------------------------------------")
    print("lr:%f" % lr)
    print("path of saving model:%s" % save_path)
    print("data root path:%s" % data_root_path)
    print("----------------------------------setting-------------------------------------")

    # save parameters to TXT.
    parameter_dict = {"datalist_json_path": datalist_json_path,
                      "data_root_path": data_root_path,
                      "seed": seed1,
                      "lr": lr,
                      "save_path": save_path,
                      'note': note}
    txt_name = 'parameter_log.txt'
    path = os.path.join(save_path, txt_name)
    with codecs.open(path, mode='a', encoding='utf-8') as file_txt:
        for key, value in parameter_dict.items():
            file_txt.write(str(key) + ':' + str(value) + '\n')

    # save this .py
    py_path_old = sys.argv[0]
    py_path_new = os.path.join(save_path, os.path.basename(py_path_old))
    shutil.copy(py_path_old, py_path_new)

    # logging
    logging.basicConfig(filename=os.path.join(model_dir_root_path, exp_name, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(parameter_dict)

    # tensorboardX
    writer = SummaryWriter(log_dir=save_path)

    # label_dict
    label_list = [i for i in range(2)]

    # patch size
    patch_size = (256, 64, 48)

    """ data generator """
    # load all data path
    with open(datalist_json_path) as f:
        datalist = json.load(f)
    train_list = datalist['train']
    val_list = datalist['val']

    # dataset
    # training
    volume_key_list = ['volume1', 'volume2']
    train_dataset = BC(train_list, data_root_path, [img1_prefix, img2_prefix], label_prefix,
                       transform=transforms.Compose([RandomRotateTransform(angle_range=(-10, 10), p_per_sample=0.2,
                                                                           volume_key_list=volume_key_list),
                                                     ScaleTransform(zoom_range=(0.7, 1.4), p_per_sample=0.2,
                                                                    volume_key_list=volume_key_list),
                                                     GaussianNoiseTransform(p_per_sample=0.15,
                                                                            volume_key_list=volume_key_list),
                                                     GaussianBlurTransform(different_sigma_per_channel=False,
                                                                           blur_sigma=(0.1, 0.3), p_per_sample=0.2,
                                                                           volume_key_list=volume_key_list),
                                                     BrightnessMultiplicativeTransform(multiplier_range=(0.70, 1.3),
                                                                                       per_channel=False,
                                                                                       p_per_sample=0.15,
                                                                                       volume_key_list=volume_key_list),
                                                     ContrastAugmentationTransform(contrast_range=(0.65, 1.5),
                                                                                   per_channel=False, p_per_sample=0.15,
                                                                                   volume_key_list=volume_key_list),
                                                     GammaTransform(gamma_range=(0.7, 1.5), retain_stats=True,
                                                                    p_per_sample=0.3,
                                                                    volume_key_list=volume_key_list),
                                                     MirrorTransform(axes=(0, 1, 2),
                                                                     volume_key_list=volume_key_list),
                                                     Crop2patchesToTensor(patch_size,
                                                                          volume_key_list=volume_key_list)]))
    val_dataset = BC(val_list, data_root_path, [img1_prefix, img2_prefix], label_prefix)

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)

    train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=10,
                                  pin_memory=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    """ model, optimizer, loss """
    model = ResUnet(in_channels=2, out_channels=2, is_dropput=False).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = dice_loss

    def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
        return initial_lr * (1 - epoch / max_epochs) ** exponent

    """ training loop """
    n_total_iter = 0
    max_epoch = max_iterations // len(train_dataloader) + 1
    best_val_dice = 0
    lr_ = lr
    model.train()
    for epoch in range(max_epoch):

        lr_ = poly_lr(epoch, max_epoch, lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        for i_batch, sampled_batch in enumerate(train_dataloader):
            # start_time
            start = time.time()

            # generate moving data
            name_batch = sampled_batch['name']
            volume_batch1 = sampled_batch['volume1'].to('cuda').float()
            volume_batch2 = sampled_batch['volume2'].to('cuda').float()
            seg_batch = sampled_batch['label']

            # ------------------
            #    Train model
            # ------------------
            # zeros the parameter gradients
            optimizer.zero_grad()

            # run model
            seg_output_list = model(torch.cat((volume_batch1, volume_batch2), dim=1))

            # Calculate loss
            # seg part
            seg_loss_list = []
            ce_loss_list = []
            dice_loss_list = []
            assert len(seg_output_list) == 4
            for index, seg_output in enumerate(seg_output_list):
                if index == 0:
                    seg_target = seg_batch.to('cuda').float()
                elif index == 1:
                    seg_target = seg_batch.numpy()
                    seg_target_bs = seg_target.shape[0]
                    seg_target = resize(seg_target.astype(float),
                                        (seg_target_bs, 1, patch_size[0]//2, patch_size[1]//2, patch_size[2]//2),
                                        order=0, mode="constant", cval=0, clip=True, anti_aliasing=False).astype(int)
                    seg_target = torch.from_numpy(seg_target).to('cuda').float()
                elif index == 2:
                    seg_target = seg_batch.numpy()
                    seg_target_bs = seg_target.shape[0]
                    seg_target = resize(seg_target.astype(float),
                                        (seg_target_bs, 1, patch_size[0]//4, patch_size[1]//4, patch_size[2]//4),
                                        order=0, mode="constant", cval=0, clip=True, anti_aliasing=False).astype(int)
                    seg_target = torch.from_numpy(seg_target).to('cuda').float()
                elif index == 3:
                    seg_target = seg_batch.numpy()
                    seg_target_bs = seg_target.shape[0]
                    seg_target = resize(seg_target.astype(float),
                                        (seg_target_bs, 1, patch_size[0]//8, patch_size[1]//8, patch_size[2]//8),
                                        order=0, mode="constant", cval=0, clip=True, anti_aliasing=False).astype(int)
                    seg_target = torch.from_numpy(seg_target).to('cuda').float()
                # elif index == 4:
                #     break

                seg_batch_std = util.standardized_seg(seg_target, label_list)
                seg_batch_one_hot = util.onehot(seg_target, label_list)

                seg_loss_ce = criterion1(seg_output, seg_batch_std)
                seg_output_softmax = F.softmax(seg_output, dim=1)
                seg_loss_dice = criterion2(seg_output_softmax[:,1:,:,:,:], seg_batch_one_hot[:,1:,:,:,:])

                seg_loss = seg_loss_ce + seg_loss_dice
                seg_loss_list.append(seg_loss)
                ce_loss_list.append(seg_loss_ce)
                dice_loss_list.append(seg_loss_dice)

            loss = 8/15 * seg_loss_list[0] + 4/15 * seg_loss_list[1] + 2/15 * seg_loss_list[2] + 1/15 * seg_loss_list[3]
            loss_CE = 8/15 * ce_loss_list[0] + 4/15 * ce_loss_list[1] + 2/15 * ce_loss_list[2] + 1/15 * ce_loss_list[3]
            loss_DICE = 8/15 * dice_loss_list[0] + 4/15 * dice_loss_list[1] + 2/15 * dice_loss_list[2] + 1/15 * dice_loss_list[3]

            # backwards and optimize
            loss.backward()
            optimizer.step()

            # ---------------------
            #     Print log
            # ---------------------
            n_total_iter += 1
            # Determine approximate time left
            end = time.time()
            iter_left = (max_epoch - epoch) * (len(train_dataloader) - i_batch)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))
            used_time = datetime.timedelta(seconds=(end - start))

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [Train index: %2d/%d] [name: %s] "
                         "[loss: %f] [used time: %s.%ss] [ETA: %s]"
                         % (epoch, max_epoch, n_total_iter, i_batch + 1,
                            len(train_dataloader), sampled_batch['name'], loss.item(),
                            used_time.seconds, used_time.microseconds // 1000, time_left))

            # tensorboardX log writer
            writer.add_scalar("loss/Total", loss.item(),          global_step=n_total_iter)
            writer.add_scalar("loss/Dice",  loss_DICE.item(), global_step=n_total_iter)
            writer.add_scalar("loss/CE",    loss_CE.item(),   global_step=n_total_iter)
            writer.add_scalar("lr", lr_, global_step=n_total_iter)


            if n_total_iter % 100 == 0:
                image = seg_batch[1, 0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 8, normalize=False)
                writer.add_image('Groundtruth', grid_image, n_total_iter)

                image = torch.argmax(seg_output_list[0], dim=1)[1:2, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 8, normalize=False)
                writer.add_image('Prediction', grid_image, n_total_iter)

                image = volume_batch1[1, 0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 8, normalize=True)
                writer.add_image('Image1', grid_image, n_total_iter)

                image = volume_batch2[1, 0:1, :, :, ::2].permute(3, 0, 2, 1).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 8, normalize=True)
                writer.add_image('Image2', grid_image, n_total_iter)

            # save model
            if n_total_iter % n_save_iter == 0:
                # Save model checkpoints
                torch.save(model.state_dict(), "%s/model_%d.pth" % (save_path, n_total_iter))
                logging.info("save model : %s/model_%d.pth" % (save_path, n_total_iter))

            # validate
            if n_total_iter % n_val_iter == 0:
                model.eval()
                logging.info('evaluating:')
                val_dice_score = 0
                val_pre_score = 0
                val_sen_score = 0
                for val_sample in tqdm(val_dataloader):
                    val_input1 = val_sample['volume1'].cpu().detach().numpy()
                    val_input2 = val_sample['volume2'].cpu().detach().numpy()
                    val_label = val_sample['label'].cpu().detach().numpy()

                    prediction, score_map = test_single_case(model, [val_input1, val_input2],
                                                             int(patch_size[0]//1.5), int(patch_size[1]//1.5),
                                                             int(patch_size[2]//1.5), patch_size, num_classes=2)

                    single_case_dc = metric.binary.dc(prediction, val_label)
                    single_case_pre = metric.binary.precision(prediction, val_label)
                    single_case_sen = metric.binary.sensitivity(prediction, val_label)

                    val_dice_score += single_case_dc
                    val_pre_score += single_case_pre
                    val_sen_score += single_case_sen

                val_dice_score /= len(val_dataloader)
                val_pre_score /= len(val_dataloader)
                val_sen_score /= len(val_dataloader)
                logging.info("evaluation result: Dice=%.4f Precision=%.4f Sensitivity=%.4f" %
                             (val_dice_score, val_pre_score, val_sen_score))
                writer.add_scalar('val_result/Dice', val_dice_score, global_step=n_total_iter)
                writer.add_scalar('val_result/Precision', val_pre_score, global_step=n_total_iter)
                writer.add_scalar('val_result/Sensitivity', val_sen_score, global_step=n_total_iter)
                model.train()

                if val_dice_score > best_val_dice:
                    best_val_iter = n_total_iter
                    best_val_dice = val_dice_score
                    torch.save(model.state_dict(), "%s/model_best.pth" % save_path)
                    logging.info("saving best model -- iteration number:%d" % best_val_iter)
                    writer.add_scalar('best_model', best_val_dice, best_val_iter)

            if n_total_iter >= max_iterations:
                break
        if n_total_iter >= max_iterations:
            break

    torch.save(model.state_dict(), "%s/model_%d.pth" % (save_path, n_total_iter))
    logging.info("save model : %s/model_%d.pth" % (save_path, n_total_iter))
    writer.close()


if __name__ == "__main__":

    train(**vars(arg))

