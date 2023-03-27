import random
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter


class GaussianNoiseTransform(object):
    """Adds additive Gaussian Noise

    Args:
        noise_variance (tuple of float): samples variance of Gaussian distribution from this interval

    CAREFUL: This transform will modify the value range of your data!
    """

    def __init__(self, noise_variance=(0, 0.1), volume_key='volume', p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.noise_variance = noise_variance
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        if np.random.uniform() < self.p_per_sample:
            volume = self.augment_gaussian_noise(volume, self.noise_variance)
        sample[self.volume_key] = volume
        return sample

    def augment_gaussian_noise(self, data_sample, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:
            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)
        return data_sample


class GaussianBlurTransform(object):
    def __init__(self, blur_sigma=(1, 5), different_sigma_per_channel=True,
                 p_per_channel=1, volume_key='volume', p_per_sample=1.0):
        """
        :param blur_sigma:
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.blur_sigma = blur_sigma
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        if np.random.uniform() < self.p_per_sample:
            volume = self.augment_gaussian_blur(volume, self.blur_sigma,  self.different_sigma_per_channel, self.p_per_channel)
        sample[self.volume_key] = volume
        return sample

    def augment_gaussian_blur(self, data_sample, sigma_range, per_channel=True, p_per_channel=1):
        def get_range_val(value, rnd_type="uniform"):
            if isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 2:
                    if value[0] == value[1]:
                        n_val = value[0]
                    else:
                        orig_type = type(value[0])
                        if rnd_type == "uniform":
                            n_val = random.uniform(value[0], value[1])
                        elif rnd_type == "normal":
                            n_val = random.normalvariate(value[0], value[1])
                        n_val = orig_type(n_val)
                elif len(value) == 1:
                    n_val = value[0]
                else:
                    raise RuntimeError("value must be either a single vlaue or a list/tuple of len 2")
                return n_val
            else:
                return value
        if not per_channel:
            sigma = get_range_val(sigma_range)
            data_sample = gaussian_filter(data_sample, sigma, order=0)
            return data_sample
        for c in range(data_sample.shape[-1]):
            if np.random.uniform() <= p_per_channel:
                if per_channel:
                    sigma = get_range_val(sigma_range)
                data_sample[:,:,c] = gaussian_filter(data_sample[:,:,c], sigma, order=0)
        return data_sample


class BrightnessMultiplicativeTransform(object):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, volume_key='volume', p_per_sample=1.0):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        if np.random.uniform() < self.p_per_sample:
            volume = self.augment_brightness_multiplicative(volume, self.multiplier_range, self.per_channel)
        sample[self.volume_key] = volume
        return sample

    def augment_brightness_multiplicative(self, data_sample, multiplier_range=(0.5, 2), per_channel=True):
        multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
        if not per_channel:
            data_sample *= multiplier
        else:
            for c in range(data_sample.shape[-1]):
                multiplier = np.random.uniform(multiplier_range[0], multiplier_range[1])
                data_sample[:, :, c] *= multiplier
        return data_sample


class GammaTransform(object):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, retain_stats=False,
                 volume_key='volume', p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        if np.random.uniform() < self.p_per_sample:
            volume = self.augment_gamma(volume, self.gamma_range, self.invert_image,
                                        per_channel=self.per_channel, retain_stats=self.retain_stats)
            sample[self.volume_key] = volume
        return sample

    def augment_gamma(self, data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                      retain_stats=False):
        if invert_image:
            data_sample = - data_sample
        if not per_channel:
            if retain_stats:
                mn = data_sample.mean()
                sd = data_sample.std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample.min()
            rnge = data_sample.max() - minm
            data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
            if retain_stats:
                data_sample = data_sample - data_sample.mean() + mn
                data_sample = data_sample / (data_sample.std() + 1e-8) * sd
        else:
            for c in range(data_sample.shape[-1]):
                if retain_stats:
                    mn = data_sample[:, :, c].mean()
                    sd = data_sample[:, :, c].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data_sample[:, :, c].min()
                rnge = data_sample[:, :, c].max() - minm
                data_sample[:, :, c] = np.power(((data_sample[:, :, c] - minm) / float(rnge + epsilon)), gamma) * float(
                    rnge + epsilon) + minm
                if retain_stats:
                    data_sample[:, :, c] = data_sample[:, :, c] - data_sample[:, :, c].mean() + mn
                    data_sample[:, :, c] = data_sample[:, :, c] / (data_sample[c].std() + 1e-8) * sd
        if invert_image:
            data_sample = - data_sample
        return data_sample


class ContrastAugmentationTransform(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True,
                 volume_key='volume', p_per_sample=1.0):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.volume_key = volume_key

    def __call__(self, sample):
        volume = sample[self.volume_key]
        if np.random.uniform() < self.p_per_sample:
            volume = self.augment_contrast(volume, contrast_range=self.contrast_range,
                                           preserve_range=self.preserve_range, per_channel=self.per_channel)
        sample[self.volume_key] = volume
        return sample

    def augment_contrast(self, data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
        if not per_channel:
            mn = data_sample.mean()
            if preserve_range:
                minm = data_sample.min()
                maxm = data_sample.max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample = (data_sample - mn) * factor + mn
            if preserve_range:
                data_sample[data_sample < minm] = minm
                data_sample[data_sample > maxm] = maxm
        else:
            for c in range(data_sample.shape[-1]):
                mn = data_sample[:, :, c].mean()
                if preserve_range:
                    minm = data_sample[:, :, c].min()
                    maxm = data_sample[:, :, c].max()
                if np.random.random() < 0.5 and contrast_range[0] < 1:
                    factor = np.random.uniform(contrast_range[0], 1)
                else:
                    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                data_sample[:, :, c] = (data_sample[:, :, c] - mn) * factor + mn
                if preserve_range:
                    data_sample[:, :, c][data_sample[:, :, c] < minm] = minm
                    data_sample[:, :, c][data_sample[:, :, c] > maxm] = maxm
        return data_sample


class RandomRotateTransform(object):
    def __init__(self, angle_range=(-10, 10), volume_key='volume', label_key='label', p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.angle_range = angle_range
        self.volume_key = volume_key
        self.label_key = label_key

    def __call__(self, sample):
        volume, label = sample[self.volume_key], sample[self.label_key]
        if np.random.uniform() < self.p_per_sample:
            rand_angle = np.random.randint(self.angle_range[0], self.angle_range[1])
            volume = rotate(volume, angle=rand_angle, axes=(1, 0), reshape=False, order=1)
            label = rotate(label, angle=rand_angle, axes=(1, 0), reshape=False, order=0)
        sample[self.volume_key], sample[self.label_key] = volume, label

        return sample


class ScaleTransform(object):
    def __init__(self, zoom_range=(0.8, 1.3), volume_key='volume', label_key='label', p_per_sample=1.0):
        self.p_per_sample = p_per_sample
        self.zoom_range = zoom_range
        self.volume_key = volume_key
        self.label_key = label_key

    def __call__(self, sample):
        volume, label = sample['volume'], sample['label']
        if np.random.uniform() < self.p_per_sample:
            zoom_factor = np.random.randint(self.zoom_range[0]*10, self.zoom_range[1]*10) / 10
            volume = zoom(volume, zoom_factor, order=1)
            label = zoom(label, zoom_factor, order=0)
        sample['volume'], sample['label'] = volume, label

        return sample


class MirrorTransform(object):
    def __init__(self, axes=(0, 1, 2), volume_key='volume', label_key='label'):
        self.volume_key = volume_key
        self.label_key = label_key
        self.axes = axes

    def __call__(self, sample):
        volume, label = sample[self.volume_key], sample[self.label_key]
        if isinstance(self.axes, int):
            if np.random.uniform() < 0.5:
                volume = np.flip(volume, self.axes)
                label = np.flip(label, self.axes)
        else:
            for axis in self.axes:
                if np.random.uniform() < 0.5:
                    volume = np.flip(volume, axis=axis)
                    label = np.flip(label, axis=axis)
        sample[self.volume_key], sample[self.label_key] = volume, label
        return sample