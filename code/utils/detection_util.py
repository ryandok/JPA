import os
import xlwt
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import label
from evaluation.SurfaceDice import compute_surface_distances, compute_robust_hausdorff, compute_average_surface_distance


def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size


def detection_sens_fp(prediction_path_list):
    # all case result value
    all_cases_result = {}
    all_cases_fp_num = 0
    all_cases_tumors_num = 0
    all_cases_tumors_hit_num = 0

    for prediction_path in tqdm(prediction_path_list):
        # single case result value
        single_result = {}
        FP_num = 0
        hit_num = 0

        # load target
        target_path = prediction_path.replace('pred', 'gt')
        target_nib = nib.load(target_path)
        spacing = target_nib.header.get_zooms()
        volume_per_voxel = float(np.prod(spacing, dtype=np.float64))
        target = target_nib.get_fdata()
        # if size < 125, do not count as tumor
        target, _, _ = remove_all_but_the_largest_connected_component(target, [1], volume_per_voxel, {1: 125})
        # cal how many tumors
        # object id range(1,.....)
        # 0 is background
        target_map, target_object_num = label(target.astype(int))
        all_cases_tumors_num += target_object_num
        target_object_size_dict = {}
        for object_id in range(1, target_object_num+1):
            target_object_size = (target_map == object_id).sum() * volume_per_voxel
            target_object_size_dict[object_id] = target_object_size

        # load prediction
        prediction = nib.load(prediction_path).get_fdata()
        # remove small connected component
        # if size < 30, do not count as tumor_pred
        prediction, _, _ = remove_all_but_the_largest_connected_component(prediction, [1], volume_per_voxel, {1: 30})
        # cal how many tumors of prediction
        prediction_map, prediction_object_num = label(prediction.astype(int))

        # {target_object_id: [prediction_object_id, ... ](maybe could have more than one prediction_object id)}
        hit_dict = {}
        # cal hit_num
        # one target object -> all prediction object
        for target_object_id in range(1, target_object_num+1):
            hit_dict[target_object_id] = []
            target_single_object = (target_map == (target_object_id)).astype(np.float64)
            hit_flag = False
            for prediction_object_id in range(1, prediction_object_num+1):
                prediction_single_object = (prediction_map == (prediction_object_id)).astype(np.float64)
                dice = metric.binary.dc(prediction_single_object, (target_single_object*prediction_single_object).astype(np.float64))
                # if dice > threshold, prediction object match to target object
                # and means hit the target object successfully
                if dice > 0:
                    hit_flag = True
                    hit_dict[target_object_id].append(prediction_object_id)
            if hit_flag:
                hit_num += 1
        all_cases_tumors_hit_num += hit_num

        # cal FP_num
        # if prediction_object_id not in hit_dict, means it is a FP
        for prediction_object_id in range(1, prediction_object_num+1):
            is_FP = True
            for hit_list in hit_dict.values():
                if prediction_object_id in hit_list:
                    is_FP = False
                    break
            if is_FP:
                FP_num += 1
                all_cases_fp_num += 1

        for target_object_id, prediction_object_list in hit_dict.items():
            target_single_object = (target_map == target_object_id).astype(np.float64)
            prediction_match_target_object = np.zeros_like(target_single_object)
            for prediction_object_id in prediction_object_list:
                prediction_single_object = (prediction_map == prediction_object_id).astype(np.float64)
                prediction_match_target_object += prediction_single_object
            if prediction_match_target_object.max() > 1:
                raise ValueError(os.path.basename(prediction_path) + ': prediction_match_target_object.max() > 1')

        if hit_num != target_object_num:
            print('missed: ', os.path.basename(prediction_path))
            single_result['Hit'] = False
        else:
            single_result['Hit'] = True
        single_result['P_num'] = target_object_num
        single_result['TP_num'] = hit_num
        single_result['FP_num'] = FP_num
        all_cases_result[os.path.basename(prediction_path).replace('pred.nii.gz', '')] = single_result

    sensitivity = round(all_cases_tumors_hit_num / all_cases_tumors_num, 4)

    return all_cases_result, sensitivity, all_cases_fp_num


def write_sens_fp_to_xls(cases_result:dict, sensitivity, all_cases_fp_num, save_path, filename):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('result', cell_overwrite_ok=True)
    style = xlwt.XFStyle()
    al = xlwt.Alignment()
    al.horz = 0x02
    al.vert = 0x01
    style.alignment = al
    worksheet.write(0, 0, 'name', style)
    worksheet.write(0, 1, 'P_num', style)
    worksheet.write(0, 2, 'TP_num', style)
    worksheet.write(0, 3, 'Hit', style)
    worksheet.write(0, 4, 'FP_num', style)

    i = 1
    for key, value in cases_result.items():
        worksheet.write(i, 0, key, style)
        worksheet.write(i, 1, value['P_num'], style)
        worksheet.write(i, 2, value['TP_num'], style)
        worksheet.write(i, 3, str(value['Hit']), style)
        worksheet.write(i, 4, value['FP_num'], style)
        i += 1

    i += 1
    worksheet.write(i+1, 0, 'Sensitivity', style)
    worksheet.write(i+1, 1, sensitivity, style)

    i += 1
    worksheet.write(i + 1, 0, 'FP num', style)
    worksheet.write(i + 1, 1, all_cases_fp_num, style)

    workbook.save(os.path.join(save_path, 'sens_fp_result_%s.xls' % filename))


