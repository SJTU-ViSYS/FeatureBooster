import os

import numpy as np

import torch
from torch.utils.data import Dataset


class MegaCOCO(Dataset):
    """
    Dataset with pre-extracted features of MegaDepth and COCO
    """
    def __init__(
        self,
        feature,
        base_path='/media/SSD/data/ms_coco',
        train=True
    ):
        self.base_path = base_path

        self.train = train

        if 'orb' in feature.lower():
            self.feature = 'orb'
        elif 'sift' in feature.lower():
            self.feature = 'sift'
        elif 'superpoint' in feature.lower():
            self.feature = 'superpoint'
        elif 'alike' in feature.lower():
            self.feature = 'alike'
        else:
            raise Exception('Not supported feature: "%s".' % feature)

        self.dataset = []
        if isinstance(base_path, list):
            for base in base_path:
                dataset_path = os.path.join(base, 'train' if train else 'val', self.feature)
                for file in os.listdir(dataset_path):
                    self.dataset.append(os.path.join(dataset_path, file))
        else:
            dataset_path = os.path.join(base_path, 'train' if train else 'val', self.feature)
            for file in os.listdir(dataset_path):
                self.dataset.append(os.path.join(dataset_path, file))

    def shuffle_dataset(self):
        print('Shuffling the training dataset...')
        np.random.shuffle(self.dataset)
        print('Done!')

    def __getitem__(self, idx):
        pair_data = np.load(self.dataset[idx])
        descs1 = pair_data['descs1']
        descs2 = pair_data['descs2']
        if 'orb' == self.feature:
            descs1 = np.unpackbits(descs1, axis=1, bitorder='little')
            descs2 = np.unpackbits(descs2, axis=1, bitorder='little')

        return {
            'kps1': torch.from_numpy(pair_data['kps1'].astype(np.float32)),
            'normalized_kps1': torch.from_numpy(pair_data['normalized_kps1'].astype(np.float32)),
            'descs1': torch.from_numpy(descs1.astype(np.float32)),
            'pos_dist1': torch.from_numpy(pair_data['pos_dist1'].astype(np.float16)),
            'ids1': torch.from_numpy(pair_data['ids1']),
            'kps2': torch.from_numpy(pair_data['kps2'].astype(np.float32)),
            'normalized_kps2': torch.from_numpy(pair_data['normalized_kps2'].astype(np.float32)),
            'descs2': torch.from_numpy(descs2.astype(np.float32)),
            'pos_dist2': torch.from_numpy(pair_data['pos_dist2'].astype(np.float16)),
            'ids2': torch.from_numpy(pair_data['ids2']),
        }

    def __len__(self):
        return len(self.dataset)
