import cv2
import numpy as np
from imgaug import augmenters as iaa
from lib.utils import dict_update

class photometricAug:
    def __init__(self, **config):
        from numpy.random import randint

        if config['enable']:
            params = config['params']
            aug_all = []

            if params.get('random_brightness', False):
                change = params['random_brightness']['max_abs_change']
                aug = iaa.Sometimes(0.5, iaa.Add((-change, change)))
                aug_all.append(aug)

            if params.get('random_contrast', False):
                change = params['random_contrast']['strength_range']
                aug = iaa.Sometimes(0.5, iaa.LinearContrast((change[0], change[1])))
                aug_all.append(aug)

            if params.get('additive_gaussian_noise', False):
                change = params['additive_gaussian_noise']['stddev_range']
                aug = iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(change[0], change[1])))
                aug_all.append(aug)

            if params.get('additive_speckle_noise', False):
                change = params['additive_speckle_noise']['prob_range']
                aug = iaa.Sometimes(0.5, iaa.ImpulseNoise(p=(change[0], change[1])))
                aug_all.append(aug)

            if params.get('motion_blur', False):
                change = params['motion_blur']['max_kernel_size']
                aug = iaa.Sometimes(0.5, iaa.MotionBlur(change))
                aug_all.append(aug)

            if params.get('GaussianBlur', False):
                change = params['GaussianBlur']['sigma']
                aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(change)))
                aug_all.append(aug)

            self.aug = iaa.Sequential(aug_all)

            if params.get('additive_shade', False):
                self.isAddShade = True
                self.shader = shadeAddor(**params['additive_shade'])
            else:
                self.isAddShade = False

            if params.get('gamma_correction', False):
                change = params['gamma_correction']['strength_range']
                self.isGamma = True
                self.gamma = iaa.Sometimes(0.5, iaa.GammaContrast((change[0], change[1])))
            else:
                self.isGamma = False
        else:
            self.aug = iaa.Sequential([
                iaa.Noop(),
            ])
            self.isAddShade = False
            self.isGamma = False

    def __call__(self, img):
        """
        image photometric augment

        Arguments:
            img: ndarray, shape (H, W), dtype uint8

        Returns: 
            aug_img: ndarray, shape (H, W), dtype uint8
        """

        aug_img = self.aug(image=img)
        if self.isAddShade:
            aug_img = self.shader(aug_img)
        aug_img = aug_img.squeeze().astype(np.uint8)
        if self.isGamma:
            aug_img = self.gamma(image=aug_img)
        return aug_img


class shadeAddor:
    default_config = {
        'nb_ellipses': 20,
        'max_scale': 0.8,
        'kernel_size_range': [250, 350]
    }

    def __init__(self, **config):
        self.config = self.default_config
        self.config = dict_update(self.config, config)

    def __call__(self, img):
        """
        image photometric augment

        Arguments:
            img: ndarray, shape (H, W), dtype uint8

        Returns: 
            shaded_img: ndarray, shape (H, W), dtype uint8
        """

        if np.random.rand() > 0.5:
            shaded_img = self.additive_shade(img, **self.config)
            return shaded_img
        else:
            return img

    def additive_shade(self, img, nb_ellipses=20, max_scale=0.8,
                kernel_size_range=[250, 350]):
        img = img[:,:,np.newaxis]
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.float32)

        for i in range(nb_ellipses):
            mask_ = np.zeros(img.shape[:2], np.uint8)
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            color = np.random.rand() * 255
            cv2.ellipse(mask_, (x, y), (ax, ay), angle, 0, 360, color, -1)
            if np.random.rand() > 0.5:
                mask += mask_.astype(np.float32)
            else:
                mask -= mask_.astype(np.float32)

        scale = np.random.rand() * max_scale
        mask = np.clip(mask, -255 * scale, 255 * scale)

        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - mask[..., np.newaxis] / 255.)
        shaded = np.clip(shaded, 0, 255)
        return shaded