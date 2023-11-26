import os
import cv2
import argparse
from tqdm import tqdm

import numpy as np

import h5py

import torch

import sys
sys.path.append('..')
from dog import Dog
from lib.utils import *
from lib.photometric import photometricAug
from lib.homographic import homographicAug

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

config = {
    'augmentation': {             
        'photometric': {
            'enable': True,
            'params': {
                'random_brightness': {
                    'max_abs_change': 50
                },
                'random_contrast': {
                    'strength_range': [0.5, 1.5]
                },
                'additive_gaussian_noise': {
                    'stddev_range': [0, 10]
                },
                'additive_shade': {
                    'max_scale': 0.8,
                    'kernel_size_range': [100, 150]
                },
                'motion_blur': {
                    'max_kernel_size': 3
                },
                'gamma_correction': {
                    'strength_range': [0.5, 2.0]
                }
            }
        },
        'homographic': {
            'enable': True,
            'params': {
                'translation': True,
                'rotation': True,
                'scaling': True,
                'perspective': True,
                'scaling_amplitude': 0.5,
                'perspective_amplitude_x': 0.2,
                'perspective_amplitude_y': 0.2,
                'patch_ratio_range': [0.7, 1.0],
                'max_angle': 3.14,
                'shift': -1,
                'allow_artifacts': True,
            },
            'valid_border_margin': 3
        }
    }
}

def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser(description="The script for pre-extract features from Megadepth")

    parser.add_argument(
        '--descriptor', type=str, required=True,
        help='type of descriptor'
    )

    parser.add_argument(
        '--dataset_name', type=str, required=True,
        help='coco or megadepth'
    )

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )

    parser.add_argument(
        '--scene_info_path', type=str, required=False,
        help='path to the processed scenes (only use for MegaDepth)'
    )

    parser.add_argument(
        '--data_type', type=str, required=True,
        help='train or val'
    )

    parser.add_argument(
        '--output_path', type=str, required=True,
        help='path for saving output'
    )

    parser.add_argument(
        '--num_kps', type=int, required=True,
        help='number of keypoint to extract'
    )

    parser.add_argument(
        '--matches_ratio', type=float, default=0.025,
        help='matches / keypoints'
    )

    parser.add_argument(
        '--gpu_id', type=str, default='0',
        help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    args = parser.parse_args()
    print(args)

    return args


def extract(feature, extractor, img, num_kps, is_gray=False):
    if 'alike' == feature.lower():
        if is_gray:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred = extractor(rgb, sub_pixel=True)
        keypoints = pred['keypoints']
        if keypoints.shape[0] <= 1:
            raise EmptyTensorError
        descriptors = pred['descriptors']
        scores = pred['scores']
        keypoints = np.hstack((keypoints, np.expand_dims(scores, 1)))
    else:
        if not is_gray:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if 'superpoint' == feature.lower():
            image = (image.astype('float32') / 255.)
            keypoints, descriptors, _ = extractor.run(image)
            if keypoints.shape[1] <= 1:
                raise EmptyTensorError
            keypoints, descriptors = keypoints.T, descriptors.T
        elif feature.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
            image = (image.astype('float32') / 255.)
            keypoints, scores, descriptors = extractor.detectAndCompute(image)
            if keypoints.shape[0] <= 1:
                raise EmptyTensorError
        elif 'orb' == feature.lower():
            kps_tuples, descriptors = extractor.detectAndCompute(image)
            # convert keypoints 
            keypoints = [cv2.KeyPoint(*kp) for kp in kps_tuples]
            if len(keypoints) <= 1:
                raise EmptyTensorError
            keypoints = np.array(
                [[kp.pt[0], kp.pt[1], kp.size / 31, np.deg2rad(kp.angle)] for kp in keypoints], 
                dtype=np.float32
            )

    if keypoints.shape[0] > num_kps:
        keypoints, descriptors = keypoints[:num_kps], descriptors[:num_kps]

    return keypoints.astype(np.float32), descriptors


def check_coco(kps1, kps2, homography1, homography2, matches_ratio):
    pos_radius = 3
    kps1_pos = torch.from_numpy(kps1[:, :2].T).cuda()
    kps2_pos = torch.from_numpy(kps2[:, :2].T).cuda()
    homo1 = torch.from_numpy(homography1.astype(np.float32)).view(3, 3).cuda()
    homo2 = torch.from_numpy(homography2.astype(np.float32)).view(3, 3).cuda()

    # Find kps1 correspondences
    kps1_warp_pos = warpPerspective(kps1_pos, homo1)
    pos_dist1 = torch.max(
        torch.abs(
            kps1_warp_pos.unsqueeze(2).float() -
            kps2_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist1 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps1.shape[0]) < matches_ratio:
        raise EmptyTensorError

    # Find kps2 correspondences
    kps2_warp_pos = warpPerspective(kps2_pos, homo2)
    pos_dist2 = torch.max(
        torch.abs(
            kps2_warp_pos.unsqueeze(2).float() -
            kps1_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist2 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps2.shape[0]) < matches_ratio:
        raise EmptyTensorError
    
    return pos_dist1.cpu().numpy(), pos_dist2.cpu().numpy()


def process_coco(feature, extractor, dataset_path, output_path, num_kps, matches_ratio):
    img_list = os.listdir(dataset_path)
    count = 0
    for img_name in img_list:
        image_path = os.path.join(dataset_path, img_name)

        image1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image1_ = image1.copy()

        try:
            kps1, descs1 = extract(feature, extractor, image1, num_kps, True)
        except EmptyTensorError:
            continue
        normalized_kps1 = normalize_keypoints(kps1, image1.shape)

        # apply homography transformation to image
        homoAugmentor = homographicAug(**config['augmentation']['homographic'])
        photoAugmentor = photometricAug(**config['augmentation']['photometric'])

        image2 = photoAugmentor(image1_)
        image2 = torch.tensor(image2, dtype=torch.float32) / 255.
        image2, homography1, homography2, valid_mask = homoAugmentor(image2)
        image2 = (image2.numpy() * 255.).astype(np.uint8)

        try:
            kps2, descs2 = extract(feature, extractor, image2, num_kps, True)
        except EmptyTensorError:
            continue
        normalized_kps2 = normalize_keypoints(kps2, image2.shape)

        try:
            pos_dist1, pos_dist2 = check_coco(kps1, kps2, homography1, homography2, matches_ratio)
        except EmptyTensorError:
            continue

        save_path = os.path.join(output_path, str(count) + '.npz')
        with open(save_path, 'wb') as file:
            np.savez(
                file,
                kps1=kps1,
                normalized_kps1=normalized_kps1,
                descs1=descs1,
                pos_dist1=pos_dist1.astype(np.float16),
                ids1=np.arange(0, kps1.shape[0]),
                kps2=kps2,
                normalized_kps2=normalized_kps2,
                descs2=descs2,
                pos_dist2=pos_dist2.astype(np.float16),
                ids2=np.arange(0, kps2.shape[0])
            )

        count += 1
        if count == 10000:
            break
    print("Sampled %d image pairs for COCO" % count)


def check_mega(kps1, kps2, base_path, pair_metadata, matches_ratio):
    pos_radius = 3

    # convert u, v to i, j
    kps1_pos_ = kps1[:,:2]
    kps1_pos_ = kps1_pos_[:, ::-1].copy()
    kps2_pos_ = kps2[:,:2]
    kps2_pos_ = kps2_pos_[:, ::-1].copy()
    kps1_pos = torch.from_numpy(kps1_pos_.T).cuda()
    kps2_pos = torch.from_numpy(kps2_pos_.T).cuda()

    # depth, intrinsics, pose
    depth_path1 = os.path.join(
        base_path, pair_metadata['depth_path1']
    )
    with h5py.File(depth_path1, 'r') as hdf5_file:
        depth1 = np.array(hdf5_file['/depth'])
    assert(np.min(depth1) >= 0)
    depth1 = torch.from_numpy(depth1.astype(np.float32)).cuda()
    intrinsics1 = pair_metadata['intrinsics1']
    intrinsics1 = torch.from_numpy(intrinsics1.astype(np.float32)).cuda()
    pose1 = pair_metadata['pose1']
    pose1 = torch.from_numpy(pose1.astype(np.float32)).cuda()

    depth_path2 = os.path.join(
        base_path, pair_metadata['depth_path2']
    )
    with h5py.File(depth_path2, 'r') as hdf5_file:
        depth2 = np.array(hdf5_file['/depth'])
    assert(np.min(depth2) >= 0)
    depth2 = torch.from_numpy(depth2.astype(np.float32)).cuda()
    intrinsics2 = pair_metadata['intrinsics2']
    intrinsics2 = torch.from_numpy(intrinsics2.astype(np.float32)).cuda()
    pose2 = pair_metadata['pose2']
    pose2 = torch.from_numpy(pose2.astype(np.float32)).cuda()

    # Find kps1 correspondences
    kps1_pos, kps1_warp_pos, ids1 = warp(
        kps1_pos,
        depth1, intrinsics1, pose1, (0.0, 0.0),
        depth2, intrinsics2, pose2, (0.0, 0.0)
    )
    pos_dist1 = torch.max(
        torch.abs(
            kps1_warp_pos.unsqueeze(2).float() -
            kps2_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist1 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps1.shape[0]) < matches_ratio:
        # print(ids_has_gt.sum(), kps1.shape[0])
        raise EmptyTensorError

    # Find kps2 correspondences
    kps2_pos, kps2_warp_pos, ids2 = warp(
        kps2_pos,
        depth2, intrinsics2, pose2, (0.0, 0.0),
        depth1, intrinsics1, pose1, (0.0, 0.0)
    )
    kps1_pos = torch.from_numpy(kps1_pos_.T).cuda()
    pos_dist2 = torch.max(
        torch.abs(
            kps2_warp_pos.unsqueeze(2).float() -
            kps1_pos.unsqueeze(1)
        ),
        dim=0
    )[0]
    ids_has_gt = (pos_dist2 <= pos_radius).sum(dim=1) >= 1
    if (ids_has_gt.sum() / kps2.shape[0]) < matches_ratio:
        raise EmptyTensorError
    
    return pos_dist1.cpu().numpy(), pos_dist2.cpu().numpy(), ids1.cpu().numpy(), ids2.cpu().numpy()


def process_megadepth(feature, extractor, dataset_path, scene_info_path, output_path, num_kps, matches_ratio, train):
    scenes = []
    if train:
        scene_list_path='megadepth_utils/train_scenes.txt'
    else:
        scene_list_path='megadepth_utils/valid_scenes.txt',
    with open(scene_list_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            scenes.append(line.strip('\n'))
    
    total_count = 0
    for scene in tqdm(scenes, total=len(scenes)):
        scene_path = os.path.join(
            scene_info_path, '%s.npz' % scene
        )
        if not os.path.exists(scene_path):
            continue
        scene_info = np.load(scene_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        valid = np.logical_and(
                    overlap_matrix >= 0.1,
                    overlap_matrix <= 1.0
                )
        pairs = np.vstack(np.where(valid))
        pairs_idx = [i for i in range(pairs.shape[1])]
        np.random.shuffle(pairs_idx)
        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']
        
        count = 0
        for pair_idx in pairs_idx:
            idx1 = pairs[0, pair_idx]
            image_path1 = os.path.join(dataset_path, image_paths[idx1])
            idx2 = pairs[1, pair_idx]
            image_path2 = os.path.join(dataset_path, image_paths[idx2])

            image1 = cv2.imread(image_path1)
            try:
                kps1, descs1 = extract(feature, extractor, image1, num_kps)
            except EmptyTensorError:
                continue
            normalized_kps1 = normalize_keypoints(kps1, image1.shape)
            image2 = cv2.imread(image_path2)
            try:
                kps2, descs2 = extract(feature, extractor, image2, num_kps)
            except EmptyTensorError:
                continue
            normalized_kps2 = normalize_keypoints(kps2, image2.shape)

            pair_data = {
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2]
            }
            try:
                pos_dist1, pos_dist2, ids1, ids2 = check_mega(kps1, kps2, dataset_path, pair_data, matches_ratio)
            except EmptyTensorError:
                continue

            save_path = os.path.join(output_path, str(total_count + count) + '.npz')
            with open(save_path, 'wb') as file:
                np.savez(
                    file,
                    kps1=kps1,
                    normalized_kps1=normalized_kps1,
                    descs1=descs1,
                    pos_dist1=pos_dist1.astype(np.float16),
                    ids1=ids1,
                    kps2=kps2,
                    normalized_kps2=normalized_kps2,
                    descs2=descs2,
                    pos_dist2=pos_dist2.astype(np.float16),
                    ids2=ids2
            )

            count += 1
            if count == 500:
                break
        
        total_count += count

    print("Sampled %d image pairs for MegaDepth" % total_count)


if __name__ == '__main__':
    # command line arguments
    args = parse_arguments()

    # CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        print("Using CUDA!!!")

    # check dataset
    if args.dataset_name.lower() not in ['coco', 'megadepth']:
        raise Exception('Not supported datatse: "%s".' % args.dataset_name)

    feature = args.descriptor

    # define the feature extractor
    if args.descriptor.lower() in ['sift', 'rootsift', 'sosnet', 'hardnet']:
        extractor = Dog(descriptor=args.descriptor.lower())
    elif 'orb' == args.descriptor.lower():
        extractor = ORBextractor(3000, 1.2, 8)
    elif 'superpoint' == args.descriptor.lower():
        sp_weights_path = Path(__file__).parent / "../extractors/SuperPointPretrainedNetwork/superpoint_v1.pth"
        extractor = SuperPointFrontend(weights_path=sp_weights_path, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7, cuda=use_cuda)
    elif 'alike' == args.descriptor.lower():
        extractor = ALike(**alike.configs['alike-l'], device='cuda' if use_cuda else 'cpu', top_k=-1, scores_th=0.2)
    else:
        raise Exception('Not supported descriptor: "%s".' % args.descriptor)

    # output path
    if not os.path.isdir(os.path.join(args.output_path, args.data_type)):
        os.mkdir(os.path.join(args.output_path, args.data_type))
    output_path = os.path.join(args.output_path, args.data_type, args.descriptor)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    else:
        print("Found existing folder! Please check out!!!")
        exit(-1)

    if args.dataset_name.lower() == 'coco':
        print("Processing COCO...")
        dataset_path = os.path.join(args.dataset_path, args.data_type+'2014')
        process_coco(feature, extractor, dataset_path, output_path, args.num_kps, args.matches_ratio)
    else:
        print("Processing MegaDepth...")
        process_megadepth(feature, extractor, args.dataset_path, args.scene_info_path, output_path, args.num_kps, args.matches_ratio, train = True if args.data_type == 'train' else False)

    print('Done!')