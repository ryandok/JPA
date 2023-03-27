import os
import torch
import argparse
import shutil
import json
from glob import glob
from networks.BaselineModel import ResUnet
from networks.JPAnet_woBreast import JPAnet

from utils.test_util_TumorSeg_OriginRes_woBreastPred import test_all_case_Nchannel_with_image_processing
from utils.detection_util import detection_sens_fp, write_sens_fp_to_xls
from utils.image_processing_tumorSeg import ImageProcessingTumorSegOriginRegion

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='2', help='GPU to use')
parser.add_argument('--model_type', type=str, default='iter_num', choices=['best', 'iter_num'])
parser.add_argument("--datalist_json_path", type=str, default='../data/datalist_Siemens.json', help="datalist path")
parser.add_argument('--data_root_path', type=str, default='/raid/hra/dataset/BreastCancerMRI_615/RawData/',
                    help='data path')
parser.add_argument("--img1_prefix", type=str, default='DCE-C1', help="img prefix")
parser.add_argument("--img2_prefix", type=str, default='DCE-C0', help="img prefix")
parser.add_argument("--label_prefix", type=str, default='TumorMask', help="label prefix")
parser.add_argument("--with_breast", type=bool, default=False, help="input image list with breast mask?")
parser.add_argument('--model_root_path', type=str, default='../model/', help='model root path')
parser.add_argument('--backbone', type=str, default='JPAnet_C1&C0',
                    choices=['Baseline_C1', 'Baseline_C1&C0', 'JPAnet_C1&C0'])
parser.add_argument('--exp_name', type=str, default='JPAnet_CropZeroDown0.5XY_C1&C0_DatalistSiemens',
                    help='experiment name')
parser.add_argument('--iter_num', type=int, default=60000, help='model iteration')
parser.add_argument('--save_result', type=bool, default=True, help='save result?')
parser.add_argument('--use_mirror_ensemble', type=bool, default=False, help='use mirror for ensemble?')
args = parser.parse_args()


def test_calculate_metric(model_type, model_root_path, data_root_path, img1_prefix, img2_prefix,
                          label_prefix, exp_name, iter_num, backbone, test_list, num_classes,
                          patch_size, stride_x, stride_y, stride_z, save_result, use_mirror_ensemble,
                          ImageProcessingTumorSegInstance):
    # load model
    if model_type == 'iter_num':
        model_path = os.path.join(model_root_path, exp_name, 'model_%d.pth' % iter_num)
    elif model_type == 'best':
        model_path = os.path.join(model_root_path, exp_name, 'model_%s.pth' % model_type)
    else:
        raise ValueError('model_type')

    if backbone == 'Baseline_C1':
        net = ResUnet(in_channels=1, out_channels=num_classes).cuda()
    elif backbone == 'Baseline_C1&C0':
        net = ResUnet(in_channels=2, out_channels=num_classes).cuda()
    elif backbone == 'JPAnet_C1&C0':
        net = JPAnet(in_channels=1, out_channels=num_classes).cuda()
    else:
        raise ValueError('backbone')

    print("init weight from {}".format(model_path))
    net.load_state_dict(torch.load(model_path))
    net.eval()

    # save inference path
    if model_type == 'iter_num':
        test_save_path = os.path.join('../data/inference/', exp_name, str(iter_num) + '_originRes_woBreastPred')
    elif model_type == 'best':
        test_save_path = os.path.join('../data/inference/', exp_name, model_type + '_originRes_woBreastPred')
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

    if backbone in ['Baseline_C1']:
        avg_metric = test_all_case_Nchannel_with_image_processing(net, test_list, data_root_path, [img1_prefix],
                                                                  label_prefix, exp_name,
                                                                  num_classes=num_classes,
                                                                  patch_size=patch_size,
                                                                  stride_x=stride_x, stride_y=stride_y,
                                                                  stride_z=stride_z,
                                                                  save_result=save_result,
                                                                  test_save_path=test_save_path,
                                                                  use_mirror_ensemble=use_mirror_ensemble,
                                                                  ImageProcessingTumorSegInstance=ImageProcessingTumorSegInstance)
    elif backbone in ['Baseline_C1&C0', 'JPAnet_C1&C0']:
        avg_metric = test_all_case_Nchannel_with_image_processing(net, test_list, data_root_path,
                                                                  [img1_prefix, img2_prefix], label_prefix,
                                                                  exp_name,
                                                                  num_classes=num_classes,
                                                                  patch_size=patch_size,
                                                                  stride_x=stride_x, stride_y=stride_y,
                                                                  stride_z=stride_z,
                                                                  save_result=save_result,
                                                                  test_save_path=test_save_path,
                                                                  use_mirror_ensemble=use_mirror_ensemble,
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
    patch_size = (256, 64, 48)
    stride_x = 120
    stride_y = 30
    stride_z = 20
    zoom_factor_tumorSeg = (0.5, 0.5, 1)
    ImageProcessingInstance = ImageProcessingTumorSegOriginRegion(zoom_factor=zoom_factor_tumorSeg,
                                                                  with_breast=args.with_breast)
    with open(args.datalist_json_path) as f:
        datalist = json.load(f)
    test_list = datalist['test']
    test_calculate_metric(args.model_type, args.model_root_path, args.data_root_path,
                          args.img1_prefix, args.img2_prefix,
                          args.label_prefix, args.exp_name, args.iter_num, args.backbone,
                          test_list, num_classes,
                          patch_size, stride_x, stride_y, stride_z,
                          args.save_result, args.use_mirror_ensemble,
                          ImageProcessingInstance)
