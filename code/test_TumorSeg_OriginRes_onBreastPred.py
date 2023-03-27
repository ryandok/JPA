import os
import torch
import argparse
import shutil
import json
from glob import glob
from networks.vnet import VNet
from networks.BaselineModel import ResUnet
from networks.JPAnet_wBreast import JPAnet

from utils.test_util_TumorSeg_OriginRes_onBreastPred import test_all_case_Nchannel_with_image_processing
from utils.detection_util import detection_sens_fp, write_sens_fp_to_xls
from utils.image_processing_tumorSeg import ImageProcessingTumorSegOriginRegion
from utils.image_processing_breastSeg import ImageProcessingBreastSeg

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--model_type', type=str, default='iter_num', choices=['best', 'iter_num'])
parser.add_argument("--datalist_json_path", type=str, default='../data/datalist_Siemens.json', help="datalist path")
parser.add_argument('--data_root_path', type=str, default='/raid/hra/dataset/BreastCancerMRI_615/RawData/',
                    help='data path')
parser.add_argument("--img1_prefix", type=str, default='DCE-C1', help="img prefix")
parser.add_argument("--img2_prefix", type=str, default='DCE-C0', help="img prefix")
parser.add_argument("--label_prefix", type=str, default='TumorMask', help="label prefix")
parser.add_argument("--breast_model_path", type=str, default='../model/BreastSegModel.pth', help="breast model path")
parser.add_argument('--model_root_path', type=str, default='../model/', help='model root path')
parser.add_argument("--with_breast", type=bool, default=True, help="input image list with breast mask?")
parser.add_argument('--backbone', type=str, default='JPAnet_C1&C0&Breast',
                    choices=['Baseline_C1&Breast', 'Baseline_C1&C0&Breast', 'JPAnet_C1&C0&Breast'])
parser.add_argument('--exp_name', type=str, default='JPAnet_CropZeroDown0.5XY_C1&C0&Breast_DatalistSiemens',
                    help='experiment name')
parser.add_argument('--iter_num', type=int, default=60000, help='model iteration')
parser.add_argument('--save_result', type=bool, default=True, help='save result?')
parser.add_argument('--use_mirror_ensemble', type=bool, default=False, help='use mirror for ensemble?')
args = parser.parse_args()


def test_calculate_metric(model_type, model_root_path, data_root_path, img1_prefix, img2_prefix,
                          label_prefix, breast_model_path, exp_name, iter_num, backbone,
                          test_list, num_classes, patch_stride_breast, patch_stride_tumor, save_result,
                          use_mirror_ensemble, ImageProcessingBreastSegInstance, ImageProcessingTumorSegInstance):
    # load model
    if model_type == 'iter_num':
        model_path = os.path.join(model_root_path, exp_name, 'model_%d.pth' % iter_num)
    elif model_type == 'best':
        model_path = os.path.join(model_root_path, exp_name, 'model_%s.pth' % model_type)
    else:
        raise ValueError('model_type')

    if backbone == 'Baseline_C1&Breast':
        net = ResUnet(in_channels=2, out_channels=num_classes).cuda()
    elif backbone == 'Baseline_C1&C0&Breast':
        net = ResUnet(in_channels=3, out_channels=num_classes).cuda()
    elif backbone == 'JPAnet_C1&C0&Breast':
        net = JPAnet(in_channels=2, out_channels=num_classes).cuda()
    else:
        raise ValueError('backbone')

    net_breast = VNet(n_channels=1, n_classes=num_classes, normalization='groupnorm', has_dropout=False).cuda()
    print("breast model initial weight from {}".format(breast_model_path))
    net_breast.load_state_dict(torch.load(breast_model_path))
    net_breast.eval()

    print("tumor model initial weight from {}".format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # save inference path
    if model_type == 'iter_num':
        test_save_path = os.path.join('../data/inference/', exp_name, str(iter_num) + '_originRes_onBreastPred')
    elif model_type == 'best':
        test_save_path = os.path.join('../data/inference/', exp_name, model_type + '_originRes_onBreastPred')
    else:
        raise ValueError('test_save_path')
    if use_mirror_ensemble:
        test_save_path += '_mirror_ensemble'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    print(test_save_path)

    # model
    shutil.copy(model_path, os.path.join(test_save_path, os.path.basename(model_path)))

    test_list = sorted(test_list)

    if backbone in ['Baseline_C1&Breast']:
        avg_metric = test_all_case_Nchannel_with_image_processing(net, net_breast, test_list, data_root_path,
                                                                  [img1_prefix], label_prefix, exp_name,
                                                                  num_classes=num_classes,
                                                                  patch_stride_breast=patch_stride_breast,
                                                                  patch_stride_tumor=patch_stride_tumor,
                                                                  save_result=save_result,
                                                                  test_save_path=test_save_path,
                                                                  use_mirror_ensemble=use_mirror_ensemble,
                                                                  ImageProcessingBreastSegInstance=ImageProcessingBreastSegInstance,
                                                                  ImageProcessingTumorSegInstance=ImageProcessingTumorSegInstance)
    elif backbone in ['Baseline_C1&C0&Breast', 'JPAnet_C1&C0&Breast']:
        avg_metric = test_all_case_Nchannel_with_image_processing(net, net_breast, test_list, data_root_path,
                                                                  [img1_prefix, img2_prefix], label_prefix, exp_name,
                                                                  num_classes=num_classes,
                                                                  patch_stride_breast=patch_stride_breast,
                                                                  patch_stride_tumor=patch_stride_tumor,
                                                                  save_result=save_result,
                                                                  test_save_path=test_save_path,
                                                                  use_mirror_ensemble=use_mirror_ensemble,
                                                                  ImageProcessingBreastSegInstance=ImageProcessingBreastSegInstance,
                                                                  ImageProcessingTumorSegInstance=ImageProcessingTumorSegInstance)

    print('\nCalculating the hit results...\n')
    root_path = test_save_path
    prediction_path_list = sorted(glob(root_path + '/*pred*'))
    cases_result, sensitivity, fp_num = detection_sens_fp(prediction_path_list)
    write_sens_fp_to_xls(cases_result, sensitivity, fp_num, root_path, exp_name)
    print('Sensitivity=%.4f, FP num=%d\n' % (sensitivity, fp_num))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    num_classes = 2
    patch_stride_breast = {"patch_size": (192, 192, 48),
                           "stride_x": 80,
                           "stride_y": 80,
                           "stride_z": 20}
    patch_stride_tumor = {"patch_size": (256, 64, 48),
                          "stride_x": 120,
                          "stride_y": 30,
                          "stride_z": 20}

    zoom_factor_breastSeg = (0.36, 0.36, 1)
    zoom_factor_tumorSeg = (0.5, 0.5, 1)

    ImageProcessingBreastSegInstance = ImageProcessingBreastSeg(zoom_factor=zoom_factor_breastSeg)
    ImageProcessingTumorSegInstance = ImageProcessingTumorSegOriginRegion(zoom_factor=zoom_factor_tumorSeg,
                                                                          with_breast=args.with_breast)

    with open(args.datalist_json_path) as f:
        datalist = json.load(f)
    test_list = datalist['test']
    test_calculate_metric(args.model_type, args.model_root_path, args.data_root_path,
                          args.img1_prefix, args.img2_prefix,
                          args.label_prefix, args.breast_model_path,
                          args.exp_name, args.iter_num, args.backbone, test_list, num_classes,
                          patch_stride_breast, patch_stride_tumor,
                          args.save_result, args.use_mirror_ensemble,
                          ImageProcessingBreastSegInstance, ImageProcessingTumorSegInstance)
