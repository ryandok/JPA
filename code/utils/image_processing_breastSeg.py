import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from scipy.ndimage import interpolation
from scipy.ndimage import binary_fill_holes
from utils.remove_connected_component import remove_all_but_the_largest_connected_component


def norm(data):
    smooth = 1e-5
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return (data-mean+smooth)/(std+smooth)


def crop_y_axis(volume_np, y_axis_end):
    data_norm = norm(volume_np)
    if y_axis_end == 'P':
        y = 0
        while (data_norm[:, y, :].max() < 1).all():
            y += 1
        start = y - 10
        if start < 0:
            start = 0

        crop_volume_np = volume_np[:, start:, :]
        return crop_volume_np, start
    elif y_axis_end == 'A':
        y = data_norm.shape[1] - 1
        while (data_norm[:, y, :].max() < 1).all():
            y -= 1
        end = y + 10 + 1
        if end > volume_np.shape[1]:
            end = volume_np.shape[1]
        crop_volume_np = volume_np[:, :end, :]
        return crop_volume_np, end
    else:
        raise ValueError("y_axis_end is not 'P' or 'A'!")


def resample(volume_np, is_label, zoom_factor: tuple):
    if not is_label:
        img_resized_np = interpolation.zoom(volume_np.astype(np.float32), (zoom_factor[0], zoom_factor[1], zoom_factor[2]),
                                            order=1).astype(np.float32)
    else:
        img_resized_np = interpolation.zoom(volume_np.astype(np.float32), (zoom_factor[0], zoom_factor[1], zoom_factor[2]),
                                            order=0).astype(np.uint8)
    return img_resized_np


def pre_processing(volume_np, y_axis_end, zoom_factor: tuple):
    volume_cropped, crop_index = crop_y_axis(volume_np, y_axis_end)
    volume_cropped_resample = resample(volume_cropped, False, zoom_factor)
    volume_cropped_resample_norm = norm(volume_cropped_resample)
    return volume_cropped_resample_norm, crop_index, volume_cropped.shape


def post_processing(prediction, cropped_shape, original_shape, crop_index, y_axis_end,
                    fill_hole=False, remove_but_largest=False):
    if fill_hole:
        prediction = binary_fill_holes(prediction)
    if remove_but_largest:
        prediction, _, _ = remove_all_but_the_largest_connected_component(prediction, [1], 1)
    prediction_resample = resize(prediction.astype(np.float32), cropped_shape, order=0).astype(np.uint8)
    prediction_origin_size = np.zeros(shape=original_shape, dtype=np.uint8)
    if y_axis_end == 'P':
        prediction_origin_size[:, crop_index:, :] = prediction_resample
    elif y_axis_end == 'A':
        prediction_origin_size[:, :crop_index, :] = prediction_resample
    else:
        raise ValueError("y_axis_end is not 'P' or 'A'!")
    return prediction_origin_size


class ImageProcessingBreastSeg:
    def __init__(self, zoom_factor=(0.36, 0.36, 1)):
        self.zoom_factor = zoom_factor
        self.crop_index = None
        self.cropped_shape = None
        self.y_axis_end = None
        self.origin_shape = None

    def pre_func(self, volume_np, affine):
        self.origin_shape = volume_np.shape
        self.y_axis_end = nib.aff2axcodes(affine)[1]
        volume_cropped_resample_norm, self.crop_index, self.cropped_shape = \
            pre_processing(volume_np, self.y_axis_end, self.zoom_factor)
        return volume_cropped_resample_norm

    def post_func(self, prediction, fill_hole=False, remove_but_largest=False):
        return post_processing(prediction, self.cropped_shape, self.origin_shape,
                               self.crop_index, self.y_axis_end, fill_hole, remove_but_largest)