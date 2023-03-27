import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage import interpolation
import nibabel as nib


def norm(data):
    smooth = 1e-5
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return (data - mean + smooth) / (std + smooth)


def crop_y_axis(volume_np_list, y_axis_end):
    data_norm = norm(volume_np_list[0])
    crop_volume_np_list = []

    if y_axis_end == 'P':
        y = 0
        while (data_norm[:, y, :].max() < 1).all():
            y += 1
        start = y - 10
        if start < 0:
            start = 0

        for volume_np in volume_np_list:
            crop_volume_np = volume_np[:, start:, :]
            crop_volume_np_list.append(crop_volume_np)
        return crop_volume_np_list, start

    elif y_axis_end == 'A':
        y = data_norm.shape[1] - 1
        while (data_norm[:, y, :].max() < 1).all():
            y -= 1
        end = y + 10 + 1
        if end > data_norm.shape[1]:
            end = data_norm.shape[1]

        for volume_np in volume_np_list:
            crop_volume_np = volume_np[:, :end, :]
            crop_volume_np_list.append(crop_volume_np)
        return crop_volume_np_list, end

    else:
        raise ValueError("y_axis_end is not 'P' or 'A'!")


def crop_xyz_axis(breast_mask_np, volume_np_list: list):
    x = 0
    while (breast_mask_np[x, :, :].max() < 1).all():
        x += 1
    start_x = x - 10
    if start_x < 0:
        start_x = 0

    x = breast_mask_np.shape[0] - 1
    while (breast_mask_np[x, :, :].max() < 1).all():
        x -= 1
    end_x = x + 10 + 1
    if end_x > breast_mask_np.shape[0]:
        end_x = breast_mask_np.shape[0]

    y = 0
    while (breast_mask_np[:, y, :].max() < 1).all():
        y += 1
    start_y = y - 10
    if start_y < 0:
        start_y = 0

    y = breast_mask_np.shape[1] - 1
    while (breast_mask_np[:, y, :].max() < 1).all():
        y -= 1
    end_y = y + 10 + 1
    if end_y > breast_mask_np.shape[1]:
        end_y = breast_mask_np.shape[1]

    z = 0
    while (breast_mask_np[:, :, z].max() < 1).all():
        z += 1
    start_z = z - 10
    if start_z < 0:
        start_z = 0

    z = breast_mask_np.shape[2] - 1
    while (breast_mask_np[:, :, z].max() < 1).all():
        z -= 1
    end_z = z + 10 + 1
    if end_z > breast_mask_np.shape[2]:
        end_z = breast_mask_np.shape[2]

    crop_volume_np_list = []
    for i in range(len(volume_np_list)):
        crop_volume_np_list.append((volume_np_list[i]*breast_mask_np)[start_x:end_x, start_y:end_y, start_z:end_z])
    return crop_volume_np_list, (start_x, end_x, start_y, end_y, start_z, end_z)


def resample(volume_np, is_label, zoom_factor: tuple):
    if not is_label:
        img_resized_np = interpolation.zoom(volume_np.astype(np.float32),
                                            (zoom_factor[0], zoom_factor[1], zoom_factor[2]),
                                            order=1).astype(np.float32)
    else:
        img_resized_np = interpolation.zoom(volume_np.astype(np.float32),
                                            (zoom_factor[0], zoom_factor[1], zoom_factor[2]),
                                            order=0).astype(np.uint8)
    return img_resized_np


def pre_processing_breast_region(breast_mask_np, volume_np_list, zoom_factor: tuple):
    volume_cropped_list, crop_index_tuple = crop_xyz_axis(breast_mask_np, volume_np_list)
    cropped_shape = volume_cropped_list[0].shape
    volume_cropped_resample_norm_list = []
    for volume_cropped in volume_cropped_list:
        volume_cropped_resample_norm_list.append(norm(resample(volume_cropped, False, zoom_factor)))
    return volume_cropped_resample_norm_list, crop_index_tuple, cropped_shape


def post_processing_breast_region(prediction, cropped_shape, original_shape, crop_index_tuple):
    prediction_resample = resize(prediction.astype(np.float32), cropped_shape, order=0).astype(np.uint8)
    prediction_origin_size = np.zeros(shape=original_shape, dtype=np.uint8)
    prediction_origin_size[crop_index_tuple[0]:crop_index_tuple[1], crop_index_tuple[2]:crop_index_tuple[3],
    crop_index_tuple[4]:crop_index_tuple[5]] = prediction_resample
    return prediction_origin_size


class ImageProcessingTumorSegBreastRegion:
    def __init__(self, zoom_factor=(0.5, 0.5, 1)):
        self.zoom_factor = zoom_factor
        self.crop_index_tuple = None
        self.cropped_shape = None
        self.origin_shape = None

    def pre_func(self, breast_mask_np, volume_np_list, _):
        self.origin_shape = volume_np_list[0].shape
        volume_cropped_resample_norm_list, self.crop_index_tuple, self.cropped_shape = \
            pre_processing_breast_region(breast_mask_np, volume_np_list, self.zoom_factor)
        return volume_cropped_resample_norm_list

    def post_func(self, prediction):
        return post_processing_breast_region(prediction, self.cropped_shape, self.origin_shape, self.crop_index_tuple)


def pre_processing_oirgin_region(volume_np_list, y_axis_end, zoom_factor: tuple):
    volume_cropped_list, crop_index = crop_y_axis(volume_np_list, y_axis_end)
    volume_cropped_resample_norm_list = []
    for volume_cropped in volume_cropped_list:
        volume_cropped_resample = resample(volume_cropped, False, zoom_factor)
        volume_cropped_resample_norm = norm(volume_cropped_resample)
        volume_cropped_resample_norm_list.append(volume_cropped_resample_norm)
    return volume_cropped_resample_norm_list, crop_index, volume_cropped_list[0].shape


def pre_processing_oirgin_region_with_breastLabel(volume_np_list, y_axis_end, zoom_factor: tuple):
    is_label_list = [False]*len(volume_np_list)
    is_label_list[-1] = True
    volume_cropped_list, crop_index = crop_y_axis(volume_np_list, y_axis_end)
    volume_cropped_resample_norm_list = []
    for volume_cropped, is_label in zip(volume_cropped_list, is_label_list):
        if not is_label:
            volume_cropped_resample = resample(volume_cropped, False, zoom_factor)
            volume_cropped_resample_norm = norm(volume_cropped_resample)
        else:
            volume_cropped_resample = resample(volume_cropped, True, zoom_factor)
            volume_cropped_resample_norm = 0.5 + (volume_cropped_resample-0.5)*0.8
        volume_cropped_resample_norm_list.append(volume_cropped_resample_norm)
    return volume_cropped_resample_norm_list, crop_index, volume_cropped_list[0].shape


def post_processing_origin_region(prediction, cropped_shape, original_shape, crop_index, y_axis_end):
    prediction_resample = resize(prediction.astype(np.float32), cropped_shape, order=0).astype(np.uint8)
    prediction_origin_size = np.zeros(shape=original_shape, dtype=np.uint8)
    if y_axis_end == 'P':
        prediction_origin_size[:, crop_index:, :] = prediction_resample
    elif y_axis_end == 'A':
        prediction_origin_size[:, :crop_index, :] = prediction_resample
    else:
        raise ValueError("y_axis_end is not 'P' or 'A'!")
    return prediction_origin_size


class ImageProcessingTumorSegOriginRegion:
    def __init__(self, zoom_factor=None, target_spacing=None, with_breast=False):
        if zoom_factor and target_spacing:
            raise ValueError("zoom factor and target spacing only one can be used!")
        elif not zoom_factor and not target_spacing:
            raise ValueError("zoom factor and target spacing are both None!")
        self.zoom_factor = zoom_factor
        self.target_spacing = target_spacing
        self.crop_index = None
        self.cropped_shape = None
        self.y_axis_end = None
        self.origin_shape = None
        self.with_breast = with_breast

    def pre_func(self, volume_np_list, affine):
        self.origin_shape = volume_np_list[0].shape
        self.y_axis_end = nib.aff2axcodes(affine)[1]
        if self.target_spacing:
            spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
            self.zoom_factor = 1 / (self.target_spacing / spacing)
        if not self.with_breast:
            volume_cropped_resample_norm_list, self.crop_index, self.cropped_shape = \
                pre_processing_oirgin_region(volume_np_list, self.y_axis_end, self.zoom_factor)
        else:
            volume_cropped_resample_norm_list, self.crop_index, self.cropped_shape = \
                pre_processing_oirgin_region_with_breastLabel(volume_np_list, self.y_axis_end, self.zoom_factor)
        return volume_cropped_resample_norm_list

    def post_func(self, prediction):
        return post_processing_origin_region(prediction, self.cropped_shape, self.origin_shape, self.crop_index, self.y_axis_end)