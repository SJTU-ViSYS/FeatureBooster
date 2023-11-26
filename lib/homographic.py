import cv2
import random
import numpy as np
from math import pi

import torch
import torch.nn.functional as F

class homographicAug:
    def __init__(self, use_cuda=False, **config):
        self.use_cuda = use_cuda

        self.homography = torch.eye(3)
        self.inv_homography = torch.eye(3)
        self.valid_border_margin = 0

        if config['enable']:
            self.isHomography = True
            params = config['params']
            self.homography, self.inv_homography = self.sample_homography(**params)
            self.valid_border_margin = config['valid_border_margin']
        else:
            self.isHomography = False


    def __call__(self, img):
        """
        Warp a image and a list of points with homography

        Arguments:
            img: tensor, shape (H, W), dtype float32 with value from 0 to 1
            point_labels: tensor, shape (N, 2(x, y)))

        Returns: 
            warp_img: tensor, shape (H, W), dtype float32 with value from 0 to 1
            warp_point_labels: tensor, shape (N, 2(x, y)))
        """

        if self.isHomography:
            warp_img = self.inv_warp_image(img, self.inv_homography, mode='bilinear')
            homography = self.scale_homography(self.homography, img.shape).cpu().numpy()
            inv_homography = self.scale_homography(self.inv_homography, img.shape).cpu().numpy()
        else:
            warp_img = img
        vaild_mask = self.compute_valid_mask(img.shape, self.inv_homography, erosion_radius=self.valid_border_margin)
        return warp_img, homography, inv_homography, vaild_mask


    def sample_homography(
            self, shift=-1, perspective=True, scaling=True, rotation=True, translation=True,
            n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
            perspective_amplitude_y=0.1, patch_ratio_range=[0.5, 0.85], max_angle=pi/2,
            allow_artifacts=False, translation_overflow=0.):
        """
        Sample a random valid homography

        Computes the homography transformation for image size with 2x2.
        The original patch, which is initialized with a simple half-size centered crop, is
        iteratively projected, scaled, rotated and translated.

        Arguments:
            perspective: A boolean that enables the perspective and affine transformations.
            scaling: A boolean that enables the random scaling of the patch.
            rotation: A boolean that enables the random rotation of the patch.
            translation: A boolean that enables the random translation of the patch.
            n_scales: The number of tentative scales that are sampled when scaling.
            n_angles: The number of tentatives angles that are sampled when rotating.
            scaling_amplitude: Controls the amount of scale.
            perspective_amplitude_x: Controls the perspective effect in x direction.
            perspective_amplitude_y: Controls the perspective effect in y direction.
            patch_ratio: Controls the size of the patches used to create the homography.
            max_angle: Maximum angle used in rotations.
            allow_artifacts: A boolean that enables artifacts when applying the homography.
            translation_overflow: Amount of border artifacts caused by translation.

        Returns:
            sampled homography: tensor, shape (3, 3)
        """

        # Corners of the output image
        pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
        # sample the patch ratio
        patch_ratio = random.uniform(patch_ratio_range[0], patch_ratio_range[1])
        # Corners of the input patch
        margin = (1 - patch_ratio) / 2
        pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                    [patch_ratio, patch_ratio], [patch_ratio, 0]])

        from numpy.random import uniform
        from scipy.stats import truncnorm

        # Random perspective and affine perturbations
        std_trunc = 2

        if perspective:
            if not allow_artifacts:
                perspective_amplitude_x = min(perspective_amplitude_x, margin)
                perspective_amplitude_y = min(perspective_amplitude_y, margin)
            perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)
            h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
            h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
            pts2 += np.array([[h_displacement_left, perspective_displacement],
                            [h_displacement_left, -perspective_displacement],
                            [h_displacement_right, perspective_displacement],
                            [h_displacement_right, -perspective_displacement]]).squeeze()

        # Random scaling
        # sample several scales, check collision with borders, randomly pick a valid one
        if scaling:
            scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
            scales = np.concatenate((np.array([1]), scales), axis=0)

            center = np.mean(pts2, axis=0, keepdims=True)
            scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
            if allow_artifacts:
                valid = np.arange(n_scales)  # all scales are valid except scale=1
            else:
                valid = (scaled >= 0.) * (scaled < 1.)
                valid = valid.prod(axis=1).prod(axis=1)
                valid = np.where(valid)[0]
            idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
            pts2 = scaled[idx,:,:]

        # Random translation
        if translation:
            t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
            if allow_artifacts:
                t_min += translation_overflow
                t_max += translation_overflow
            pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

        # Random rotation
        # sample several rotations, check collision with borders, randomly pick a valid one
        if rotation:
            angles = np.linspace(-max_angle, max_angle, num=n_angles)
            angles = np.concatenate((angles, np.array([0.])), axis=0)  # in case no rotation is valid
            center = np.mean(pts2, axis=0, keepdims=True)
            rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                        np.cos(angles)], axis=1), [-1, 2, 2])
            rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
            if allow_artifacts:
                valid = np.arange(n_angles)  # all scales are valid except scale=1
            else:
                valid = (rotated >= 0.) * (rotated < 1.)
                valid = valid.prod(axis=1).prod(axis=1)
                valid = np.where(valid)[0]
            idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
            pts2 = rotated[idx,:,:]

        # Rescale to the 2x2 size
        shape = np.array([2, 2])
        pts1 *= shape[np.newaxis,:]
        pts2 *= shape[np.newaxis,:]

        def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

        def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

        homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
        
        homography = np.linalg.inv(homography)
        inv_homography = np.linalg.inv(homography)
        homography = torch.tensor(homography).type(torch.FloatTensor)
        inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)
        
        return homography, inv_homography


    def scale_homography(self, homography, image_shape, shift=-1):
        """
        Scale a homography for size 2x2 to a homography for image_shape 

        Arguments:
            homography: tensor, shape (3, 3)
            image_shape: list of image shape

        Returns: 
            scaled homography: tensor, shape (3, 3)
        """

        height, width = image_shape[0], image_shape[1]
        trans = torch.tensor([[2./width, 0., shift], [0., 2./height, shift], [0., 0., 1.]])
        scale_homography = trans.inverse() @ homography @ trans
        return scale_homography


    def warp_points(self, points, homography):
        """
        Warp a list of points with the given homography

        Arguments:
            points: tensor, shape (N, 2(x, y))).
            homography: tensor, shape (3, 3)

        Returns: 
            warped points: tensor, shape (N, 2)
        """

        # expand points to (x, y, 1)
        points = torch.cat((points.float(), torch.ones((points.shape[0], 1))), dim=1)
        
        warped_points = homography @ points.transpose(0,1)
        warped_points = warped_points.transpose(0, 1)

        warped_points = warped_points[:, :2] / warped_points[:, 2:]
        return warped_points


    def inv_warp_image(self, img, inv_homography, mode='bilinear'):
        """
        Inverse warp image

        Arguments:
            img: tensor, shape (H, W)
            inv_homography: tensor, shape (3, 3)

        Returns:
            warped image: tensor, shape (H, W)
        """

        # compute inverse warped points
        H, W = img.shape
        img = img.view(1,1,img.shape[0], img.shape[1])

        coord_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)), dim=2)
        coord_cells = coord_cells.transpose(0, 1)
        if self.use_cuda:
            coord_cells = coord_cells.cuda()
        coord_cells = coord_cells.contiguous()

        src_pixel_coords = self.warp_points(coord_cells.view([-1, 2]), inv_homography)
        src_pixel_coords = src_pixel_coords.view([1, H, W, 2])
        src_pixel_coords = src_pixel_coords.float()

        warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
        warped_img = warped_img.squeeze()
        return warped_img


    def compute_valid_mask(self, image_shape, inv_homography, erosion_radius=0):
        """
        Compute a boolean mask of the valid pixels resulting from an homography applied to
        an image of a given shape. Pixels that are False correspond to bordering artifacts.
        A margin can be discarded using erosion.

        Arguments:
            image_shape: list of image shape
            inv_homography: tensor, shape (3, 3)
            erosion_radius: radius of the margin to be discarded

        Returns: tensor, shape (H, W)
        """

        mask = torch.ones(image_shape[0], image_shape[1])
        mask = self.inv_warp_image(mask, inv_homography, mode='nearest')
        mask = mask.cpu().numpy().astype(np.uint8)
        if erosion_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask # torch.tensor(mask)