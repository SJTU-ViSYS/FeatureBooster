import os
import cv2
from tqdm import tqdm

import numpy as np

import h5py

from dog import Dog
from lib.utils import *

import torch
from torch.utils.data import Dataset

import sys
from pathlib import Path

orb_path = Path(__file__).parent / "../extractors/orbslam2_features/lib"
sys.path.append(str(orb_path))
from orbslam2_features import ORBextractor

superpoint_path = Path(__file__).parent / "../extractors/SuperPointPretrainedNetwork"
sys.path.append(str(superpoint_path))
from demo_superpoint import SuperPointFrontend

alike_path = Path(__file__).parent / "../extractors/ALIKE"
sys.path.append(str(alike_path))
import alike
from alike import ALike


class MegaDepth(Dataset):
    def __init__(
        self,
        feature,
        scene_list_path='datasets/megadepth_utils/train_scenes.txt',
        scene_info_path='/local/dataset/megadepth/scene_info',
        base_path='/local/dataset/megadepth',
        train=True,
        min_overlap_ratio=.1,
        max_overlap_ratio=1,
        pairs_per_scene=200,
        kps_per_image=2048,
        crop_image_size=-1
    ):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))

        self.scene_info_path = scene_info_path
        self.base_path = base_path

        self.train = train

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio

        self.pairs_per_scene = pairs_per_scene if train else 50

        self.num_kps = kps_per_image

        self.crop_image_size = crop_image_size

        self.feature = feature.lower()

        if 'sift' in self.feature:
            self.feature_extractor = Dog(descriptor='sift')
        elif 'orb' in self.feature:
            self.feature_extractor = ORBextractor(3000, 1.2, 8)
        elif 'superpoint' in self.feature:
            sp_weights_path = Path(__file__).parent / "../extractors/SuperPointPretrainedNetwork/superpoint_v1.pth"
            self.feature_extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=torch.cuda.is_available())
        elif 'alike' in self.feature:
            self.feature_extractor = ALike(**alike.configs['alike-l'], device='cuda' if torch.cuda.is_available() else 'cpu', top_k=-1, scores_th=0.2)
        else:
            raise Exception('Not supported descriptor: "%s".' % feature)

        self.dataset = []

    def build_dataset(self):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
            print('Building the validation dataset...')
        else:
            print('Building a new training dataset...')
        for scene in tqdm(self.scenes, total=len(self.scenes)):
            scene_info_path = os.path.join(
                self.scene_info_path, '%s.npz' % scene
            )
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']

            valid = np.logical_and(
                        overlap_matrix >= self.min_overlap_ratio,
                        overlap_matrix <= self.max_overlap_ratio
                    )
            
            pairs = np.vstack(np.where(valid))
            pairs_idx = [i for i in range(pairs.shape[1])]
            np.random.shuffle(pairs_idx)
            
            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']
            
            count = 0
            for pair_idx in pairs_idx:
                if count == self.pairs_per_scene:
                    break

                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(list(
                    points3D_id_to_2D[idx1].keys() &
                    points3D_id_to_2D[idx2].keys()
                ))

                match = np.random.choice(matches)

                point2D1 = points3D_id_to_2D[idx1][match]
                point2D2 = points3D_id_to_2D[idx2][match]
                central_match = np.array([
                    point2D1[1], point2D1[0],
                    point2D2[1], point2D2[0]
                ])

                pair_data = {
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match
                }
                try:
                    (
                        kps1, normalized_kps1, descs1, bbox1,
                        kps2, normalized_kps2, descs2, bbox2
                    ) = self.extract_pair(pair_data)
                except EmptyTensorError:
                    continue
                pair_data.update({
                    'kps1': kps1,
                    'normalized_kps1': normalized_kps1,
                    'descs1': descs1,
                    'bbox1': bbox1,
                    'kps2': kps2,
                    'normalized_kps2': normalized_kps2,
                    'descs2': descs2,
                    'bbox2': bbox2
                })
                self.dataset.append(pair_data)
                count += 1
            
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def extract_feature(self, bgr):
        if 'alike' in self.feature:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pred = self.feature_extractor(rgb, sub_pixel=True)
            keypoints = pred['keypoints']
            if keypoints.shape[0] <= 1:
                raise EmptyTensorError
            descriptors = pred['descriptors']
            scores = pred['scores']
            keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if 'superpoint' in self.feature:
                image = (image.astype('float32') / 255.)
                keypoints, descriptors, _ = self.feature_extractor.run(image)
                if keypoints.shape[0] <= 1:
                    raise EmptyTensorError
                keypoints, descriptors = keypoints.T, descriptors.T
            elif 'sift' in self.feature:
                image = (image.astype('float32') / 255.)
                keypoints, scores, descriptors = self.feature_extractor.detectAndCompute(image)
                if keypoints.shape[0] <= 1:
                    raise EmptyTensorError
            elif 'orb' in self.feature:
                kps_tuples, descriptors = self.feature_extractor.detectAndCompute(image)
                # convert keypoints 
                keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
                if len(keypoints) <= 1:
                    raise EmptyTensorError
                keypoints = np.array(
                    [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
                    dtype=np.float32
                )

        if keypoints.shape[0] > self.num_kps:
            keypoints, descriptors = keypoints[:self.num_kps], descriptors[:self.num_kps]
        normalized_kps = normalize_keypoints(keypoints, bgr.shape)

        return keypoints, normalized_kps, descriptors

    def extract_pair(self, pair_metadata):
        image_path1 = os.path.join(
            self.base_path, pair_metadata['image_path1']
        )
        image1 = cv2.imread(image_path1)

        image_path2 = os.path.join(
            self.base_path, pair_metadata['image_path2']
        )
        image2 = cv2.imread(image_path2)

        if self.crop_image_size != -1:
            central_match = pair_metadata['central_match']
            image1, bbox1, image2, bbox2 = self.crop(image1, image2, central_match)
        else:
            bbox1 = np.array([0, 0])
            bbox2 = np.array([0, 0])

        kps1, normalized_kps1, descs1 = self.extract_feature(image1)
        kps2, normalized_kps2, descs2 = self.extract_feature(image2)
        if 'orb' in self.feature:
            descs1 = np.unpackbits(descs1, axis=1, bitorder='little')
            descs2 = np.unpackbits(descs2, axis=1, bitorder='little')
        # convert u, v to i, j
        kps1 = kps1[:,:2]
        kps1 = kps1[:, ::-1]
        kps2 = kps2[:,:2]
        kps2 = kps2[:, ::-1]

        return (
            kps1, normalized_kps1, descs1, bbox1,
            kps2, normalized_kps2, descs2, bbox2
        )

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(
            self.base_path, pair_metadata['depth_path1']
        )
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert(np.min(depth1) >= 0)
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(
            self.base_path, pair_metadata['depth_path2']
        )
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert(np.min(depth2) >= 0)
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        if self.crop_image_size != -1:
            bbox1 = pair_metadata['bbox1']
            bbox2 = pair_metadata['bbox2']

            depth1 = depth1[
                bbox1[0] : bbox1[0] + self.crop_image_size,
                bbox1[1] : bbox1[1] + self.crop_image_size
            ]
            depth2 = depth2[
                bbox2[0] : bbox2[0] + self.crop_image_size,
                bbox2[1] : bbox2[1] + self.crop_image_size
            ]

        return (
            depth1, intrinsics1, pose1,
            depth2, intrinsics2, pose2
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.crop_image_size // 2, 0)
        if bbox1_i + self.crop_image_size >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.crop_image_size
        bbox1_j = max(int(central_match[1]) - self.crop_image_size // 2, 0)
        if bbox1_j + self.crop_image_size >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.crop_image_size

        bbox2_i = max(int(central_match[2]) - self.crop_image_size // 2, 0)
        if bbox2_i + self.crop_image_size >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.crop_image_size
        bbox2_j = max(int(central_match[3]) - self.crop_image_size // 2, 0)
        if bbox2_j + self.crop_image_size >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.crop_image_size

        return (
            image1[
                bbox1_i : bbox1_i + self.crop_image_size,
                bbox1_j : bbox1_j + self.crop_image_size
            ],
            np.array([bbox1_i, bbox1_j]),
            image2[
                bbox2_i : bbox2_i + self.crop_image_size,
                bbox2_j : bbox2_j + self.crop_image_size
            ],
            np.array([bbox2_i, bbox2_j])
        )

    def __getitem__(self, idx):
        (
            depth1, intrinsics1, pose1,
            depth2, intrinsics2, pose2
        ) = self.recover_pair(self.dataset[idx])

        return {
            'kps1': torch.from_numpy(self.dataset[idx]['kps1'].astype(np.float32)),
            'normalized_kps1': torch.from_numpy(self.dataset[idx]['normalized_kps1'].astype(np.float32)),
            'descs1': torch.from_numpy(self.dataset[idx]['descs1'].astype(np.float32)),
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'bbox1': torch.from_numpy(self.dataset[idx]['bbox1'].astype(np.float32)),
            'kps2': torch.from_numpy(self.dataset[idx]['kps2'].astype(np.float32)),
            'normalized_kps2': torch.from_numpy(self.dataset[idx]['normalized_kps2'].astype(np.float32)),
            'descs2': torch.from_numpy(self.dataset[idx]['descs2'].astype(np.float32)),
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'pose2': torch.from_numpy(pose2.astype(np.float32)),
            'bbox2': torch.from_numpy(self.dataset[idx]['bbox2'].astype(np.float32))
        }
