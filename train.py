import os
import math
import shutil
import yaml
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.losses import *
from lib.network import *
from lib.datatsets import *
from lib.utils import *
from lib.exceptions import *


# Argument parsing
parser = argparse.ArgumentParser(description="Training FeatureBooster using MegaDepth dataset")

parser.add_argument(
    '--descriptor', type=str, required=True,
    help='descriptor to be boosted by neural network (descriptors in config file)'
)

parser.add_argument(
    '--no_kenc', dest='no_kenc', action='store_true',
    help='whether not to use keypoint encoder'
)
parser.set_defaults(no_kenc=False)

parser.add_argument(
    '--no_cross', dest='no_cross', action='store_true',
    help='whether not to use cross boosting'
)
parser.set_defaults(no_cross=False)

parser.add_argument(
    '--use_mha', dest='use_mha', action='store_true',
    help='whether to use multi-head attention'
)
parser.set_defaults(use_mha=False)

parser.add_argument(
    '--dataset_path', type=str, required=True,
    help='path to the dataset'
)

parser.add_argument(
    '--scene_info_path', type=str, required=True,
    help='path to the processed scenes'
)

parser.add_argument(
    '--config', type=str, required=True,
    help='path to the model config file'
)

parser.add_argument(
    '--num_epochs', type=int, default=50,
    help='number of training epochs'
)

parser.add_argument(
    '--optimizer', default='adamw', type=str,
    help='The optimizer to use (option: sgd and adam(w))'
)

parser.add_argument(
    '--wd', '--weight-decay', default=0, type=float,
    metavar='W', help='weight decay (default: 0)',
    dest='weight_decay'
)

parser.add_argument(
    '--lr', type=float, default=1e-3,
    help='initial learning rate'
)

parser.add_argument(
    '--lr_decay', type=str, default='cos',
    help='learning rate decay way (cos, linear, none)'
)

parser.add_argument(
    '--wo_warmup', dest='wo_warmup', action='store_true',
    help='no using warm up for learning rate'
)
parser.set_defaults(wo_warmup=False)

parser.add_argument(
    '--warmup_len', type=int, default=500,
    help='number of iteration for warm up'
)

parser.add_argument(
    '--num_workers', type=int, default=8,
    help='number of workers for data loading'
)

parser.add_argument(
    '--batch_size', type=int, default=16,
    help='batch size for data loading'
)

parser.add_argument(
    '--pairs_per_scene', type=int, default=300,
    help='number of pair for per scene in dataset'
)

parser.add_argument(
    '--crop_image_size', type=int, default=512,
    help='size of dataset image to crop'
)

parser.add_argument(
    '--lboost_weight', type=float, default=10,
    help='the weight of L_boost in loss function'
)

parser.add_argument(
    '--use_validation', dest='use_validation', action='store_true',
    help='use the validation split'
)
parser.set_defaults(use_validation=False)

parser.add_argument(
    '--initial_checkpoint', type=str, default=None,
    help='path to the initial checkpoint'
)

parser.add_argument(
    '--checkpoint_directory', type=str, default='runs',
    help='directory for training checkpoints'
)

parser.add_argument(
    '--log_interval', type=int, default=25,
    help='loss logging interval'
)

parser.add_argument(
    '--gpu_id', type=str, default='0',
    help='id(s) for CUDA_VISIBLE_DEVICES'
)

args = parser.parse_args()
print(args)

# CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    print("Using CUDA!!!")

# Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)

# Log
suffix = 'megadepth_k+_{}_bs{}_ep{}_lr{}'.format(args.descriptor, args.batch_size, args.num_epochs, args.lr)

if args.optimizer.lower() == 'sgd':
    suffix = suffix + '_sgd'
elif args.optimizer.lower() == 'adam':
    suffix = suffix + '_adam'
elif args.optimizer.lower() == 'adamw':
    suffix = suffix + '_adamw'
else:
    raise Exception('Not supported optimizer: "%s".' % args.optimizer)

if args.lr_decay.lower() == 'cos':
    suffix = suffix + '_cos'
elif args.lr_decay.lower() == 'linear':
    suffix = suffix + '_linear'
elif args.lr_decay.lower() == 'none':
    suffix = suffix
else:
    raise Exception('Not supported learning rate decay way: "%s".' % args.lr_decay)

suffix = suffix + '_lboost' + str(args.lboost_weight)

if not args.wo_warmup:
    suffix = suffix + '_warmup' + str(args.warmup_len)

if args.no_kenc:
    suffix = suffix + '_nokenc'

if args.no_cross:
    suffix = suffix + '_nocross'

if args.use_mha:
    suffix = suffix + '_mha'

if args.initial_checkpoint is not None:
    suffix = suffix + '_tune'


if not os.path.isdir(args.checkpoint_directory):
    os.mkdir(args.checkpoint_directory)
result_directory = os.path.join(args.checkpoint_directory, suffix)
if os.path.isdir(result_directory):
    print('[Warning] save directory already exists!')
else:
    os.mkdir(result_directory)
# Tensorboard writer
writer = SummaryWriter(result_directory)
iter_idx = 0
# Checkpoint directory
if os.path.isdir(os.path.join(result_directory, 'checkpoints')):
    print('[Warning] Checkpoint directory already exists!')
else:
    os.mkdir(os.path.join(result_directory, 'checkpoints'))
# Log file
if os.path.exists(os.path.join(result_directory, 'log.txt')):
    print('[Warning] Log file already exists!')
log_file = open(os.path.join(result_directory, 'log.txt'), 'a+')

# Dataset
if args.use_validation:
    validation_dataset = MegaDepth(
        feature=args.descriptor,
        scene_list_path='datasets/megadepth_utils/valid_scenes_disk.txt',
        scene_info_path=args.scene_info_path,
        base_path=args.dataset_path,
        train=False,
        pairs_per_scene=args.pairs_per_scene,
        crop_image_size=args.crop_image_size
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=coco_collate,
        pin_memory=True
    )

training_dataset = MegaDepth(
    feature=args.descriptor,
    scene_list_path='datasets/megadepth_utils/train_scenes_disk.txt',
    scene_info_path=args.scene_info_path,
    base_path=args.dataset_path,
    pairs_per_scene=args.pairs_per_scene,
    crop_image_size=args.crop_image_size
)
training_dataloader = DataLoader(
    training_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    collate_fn=coco_collate,
    pin_memory=True
)


def adjust_learning_rate(optimizer, epoch_idx):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    if  args.lr_decay.lower() == 'none':
        return

    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        lr_step = group['step']
        if not args.wo_warmup and group['step'] < args.warmup_len:
                group['lr'] = group['init_lr'] * (group['step'] + 1) / args.warmup_len
        else:
            if args.lr_decay.lower() == 'cos':
                group['lr'] = group['init_lr'] * (0.5 * (
                    1. + math.cos(math.pi * float(epoch_idx - 1) / float(args.num_epochs))))
            elif args.lr_decay.lower() == 'linear':
                group['lr'] = group['init_lr'] * (
                    1. - float(epoch_idx - 1) / float(args.num_epochs))
            else:
                raise Exception('Not supported learning rate decay way: "%s".' % args.lr_decay)
        lr = group['lr']

    writer.add_scalar('Iteration learning rate', lr, global_step=lr_step)


def process_epoch(data_loader, booster, optimizer, epoch_idx, train=True):
    losses = []
    torch.set_grad_enabled(train)

    count = []

    if train:
        booster.train()
    else:
        booster.eval()
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    for batch_idx, data in progress_bar:
        torch.cuda.empty_cache()
        match_loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
        n_valid_samples = 0

        num_ids_has_gt = 0

        origin_ap_loss = []
        boost_ap_loss = []

        b = len(data['kps1'])
        for idx_in_batch in range(b):
            # Annotations
            kps1 = data['kps1'][idx_in_batch].to(device)
            normalized_kps1 = data['normalized_kps1'][idx_in_batch].to(device)
            descs1 = data['descs1'][idx_in_batch].to(device)
            depth1 = data['depth1'][idx_in_batch].to(device)  # [h1, w1]
            intrinsics1 = data['intrinsics1'][idx_in_batch].to(device)  # [3, 3]
            pose1 = data['pose1'][idx_in_batch].view(4, 4).to(device)  # [4, 4]
            bbox1 = data['bbox1'][idx_in_batch].to(device)  # [2]

            kps2 = data['kps2'][idx_in_batch].to(device)
            normalized_kps2 = data['normalized_kps2'][idx_in_batch].to(device)
            descs2 = data['descs2'][idx_in_batch].to(device)
            depth2 = data['depth2'][idx_in_batch].to(device)
            intrinsics2 = data['intrinsics2'][idx_in_batch].to(device)
            pose2 = data['pose2'][idx_in_batch].view(4, 4).to(device)
            bbox2 = data['bbox2'][idx_in_batch].to(device)

            descs1_refine = booster((descs1 * 2.0 - 1.0), normalized_kps1) if ('orb' in args.descriptor) else booster(descs1, normalized_kps1)
            descs2_refine = booster((descs2 * 2.0 - 1.0), normalized_kps2) if ('orb' in args.descriptor) else booster(descs2, normalized_kps2)
            if '+Boost-B' in args.descriptor:
                descs1_refine = (descs1_refine + ((descs1_refine >= 0).type_as(descs1_refine) - descs1_refine).detach()) * 2.0 - 1.0
                descs2_refine = (descs2_refine + ((descs2_refine >= 0).type_as(descs2_refine) - descs2_refine).detach()) * 2.0 - 1.0

            # Get the matching distance and label
            pos_radius = 3
            neg_radius = 15
            
            # Find kps1 correspondences
            kps1_pos = kps1[:, :2].t()
            try:
                kps1_pos, kps1_warp_pos, ids = warp(
                    kps1_pos,
                    depth1, intrinsics1, pose1, bbox1,
                    depth2, intrinsics2, pose2, bbox2
                )
            except EmptyTensorError:
                continue
            descriptors1_rf = descs1_refine[ids, :]

            kps2_pos = kps2[:, :2].t()
            position_distance = torch.max(
                torch.abs(
                    kps1_warp_pos.unsqueeze(2).float() -
                    kps2_pos.unsqueeze(1)
                ),
                dim=0
            )[0]
            ids_has_gt = (position_distance <= pos_radius).sum(dim=1) >= 1
            descriptors1_rf = descriptors1_rf[ids_has_gt, :]

            if descriptors1_rf.shape[0] == 0:
                continue
            num_ids_has_gt += descriptors1_rf.shape[0]

            position_distance = position_distance[ids_has_gt]
            pos_labels = (position_distance <= pos_radius).float()
            neg_labels = (position_distance >= neg_radius).float()

            descriptors1 = descs1[ids, :]
            descriptors1 = descriptors1[ids_has_gt, :]
            if 'orb' in args.descriptor.lower():
                descriptor_distance_origin = (descriptors1.shape[1] - (descriptors1  * 2. - 1.) @ (descs2 * 2. - 1.).t()) * 0.5
                origin_ap_loss_ = FastAPLoss(num_bins=10, max_distance=descriptors1.shape[1])(descriptor_distance_origin, pos_labels, neg_labels, size_average=False)
            else:
                descriptor_distance_origin = 2 - 2 * descriptors1 @ descs2.t()
                origin_ap_loss_ = FastAPLoss(num_bins=10, max_distance=4)(descriptor_distance_origin, pos_labels, neg_labels, size_average=False)
            origin_ap_loss.append(origin_ap_loss_.view(-1))

            if '+Boost-B' in args.descriptor:
                descriptor_distance = (descriptors1_rf.shape[1] - descriptors1_rf @ descs2_refine.t()) * 0.5
                boost_ap_loss_ = FastAPLoss(num_bins=10, max_distance=descriptors1_rf.shape[1])(descriptor_distance, pos_labels, neg_labels, size_average=False)
            else:
                descriptor_distance = 2 - 2 * descriptors1_rf @ descs2_refine.t()
                boost_ap_loss_ = FastAPLoss(num_bins=10, max_distance=4)(descriptor_distance, pos_labels, neg_labels, size_average=False)
            boost_ap_loss.append(boost_ap_loss_.view(-1))

            # Find kps2 correspondences
            try:
                kps2_pos, kps2_warp_pos, ids = warp(
                    kps2_pos,
                    depth2, intrinsics2, pose2, bbox2,
                    depth1, intrinsics1, pose1, bbox1
                )
            except EmptyTensorError:
                continue
            descriptors2_rf = descs2_refine[ids, :]

            kps1_pos = kps1[:, :2].t()
            position_distance = torch.max(
                torch.abs(
                    kps2_warp_pos.unsqueeze(2).float() -
                    kps1_pos.unsqueeze(1)
                ),
                dim=0
            )[0]
            ids_has_gt = (position_distance <= pos_radius).sum(dim=1) >= 1
            descriptors2_rf = descriptors2_rf[ids_has_gt, :]

            if descriptors2_rf.shape[0] == 0:
                continue
            num_ids_has_gt += descriptors2_rf.shape[0]

            position_distance = position_distance[ids_has_gt]
            pos_labels = (position_distance <= pos_radius).float()
            neg_labels = (position_distance >= neg_radius).float()

            descriptors2 = descs2[ids, :]
            descriptors2 = descriptors2[ids_has_gt, :]
            if 'orb' in args.descriptor:
                descriptor_distance_origin = (descriptors2.shape[1] - (descriptors2  * 2. - 1.) @ (descs1 * 2. - 1.).t()) * 0.5
                origin_ap_loss_ = FastAPLoss(num_bins=10, max_distance=descriptors2.shape[1])(descriptor_distance_origin, pos_labels, neg_labels, size_average=False)
            else:
                descriptor_distance_origin = 2 - 2 * descriptors2 @ descs1.t()
                origin_ap_loss_ = FastAPLoss(num_bins=10, max_distance=4)(descriptor_distance_origin, pos_labels, neg_labels, size_average=False)
            origin_ap_loss.append(origin_ap_loss_.view(-1))

            if '+Boost-B' in args.descriptor:
                descriptor_distance = (descriptors2_rf.shape[1] - descriptors2_rf @ descs1_refine.t()) * 0.5
                boost_ap_loss_ = FastAPLoss(num_bins=10, max_distance=descriptors2_rf.shape[1])(descriptor_distance, pos_labels, neg_labels, size_average=False)
            else:
                descriptor_distance = 2 - 2 * descriptors2_rf @ descs1_refine.t()
                boost_ap_loss_ = FastAPLoss(num_bins=10, max_distance=4)(descriptor_distance, pos_labels, neg_labels, size_average=False)
            boost_ap_loss.append(boost_ap_loss_.view(-1))

            n_valid_samples += 1

        if train:
            global iter_idx
            iter_idx += 1

        if n_valid_samples == 0:
            continue

        boost_ap_loss = torch.cat(boost_ap_loss, dim=0)
        match_loss = boost_ap_loss.mean()
        if train:
            origin_ap_loss = torch.cat(origin_ap_loss, dim=0)
            origin_match_loss = origin_ap_loss.mean()
            if train:
                writer.add_scalar('Iteration origin matching loss', origin_match_loss.item(), global_step=iter_idx)
            boost_loss = torch.mean(F.relu((1. - origin_ap_loss) / (1. - boost_ap_loss) - 1.))
            loss = match_loss + args.lboost_weight * boost_loss
        else:
            loss = match_loss

        if train:
            writer.add_scalar('Iteration matching loss', match_loss.item(), global_step=iter_idx)
            writer.add_scalar('Iteration boost loss', boost_loss.item(), global_step=iter_idx)
            writer.add_scalar('Iteration training loss', loss.item(), global_step=iter_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer, epoch_idx)

        # add loss to history
        losses.append(loss.item())

        # update the progress bar
        progress_bar.set_postfix(loss=('%.5f' % loss.item()))

        # write the message of batch in the log file
        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %02d - batch %04d / %04d - Loss: %f\n' % (
                'train' if train else 'valid',
                epoch_idx, batch_idx, len(data_loader), np.mean(losses)
            ))

        count.append(num_ids_has_gt)

    print(np.mean(count))

    writer.add_scalar('%s mean loss' % ('train' if train else 'valid'), np.mean(losses), global_step=epoch_idx)

    # write the message of batch in the log file
    log_file.write('[%s] epoch %02d - avg_loss: %f\n' % ('train' if train else 'valid', epoch_idx, np.mean(losses)))
    log_file.flush()

    return np.mean(losses)


# set default tensor data type
torch.set_default_tensor_type(torch.FloatTensor)

# descriptor to be boosted
feature = args.descriptor

# load config
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# print(yaml.dump(config, indent=2))
print(config[feature])

# Model
booster = FeatureBooster(config[feature], use_kenc=not args.no_kenc, use_cross=not args.no_cross, use_mha=args.use_mha)
booster.to(device)

if args.optimizer == 'sgd':
    optimizer = optim.SGD(
        booster.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'adam':
    optimizer= optim.Adam(
        [
            {"params": booster.parameters(), "init_lr": args.lr, "lr": args.lr},
        ],
    )
elif args.optimizer == 'adamw':
    optimizer= optim.AdamW(
        [
            {"params": booster.parameters(), "init_lr": args.lr, "lr": args.lr}
        ],
        # weight_decay=args.weight_decay
    )
else:
    raise Exception('Not supported optimizer: "%s".' % config['training']['optimizer'])

# Initialize the history
train_loss_history = []
validation_loss_history = []
start_epoch = 1
# load initial checkpoint if needed
if args.initial_checkpoint is not None:
    checkpoint = torch.load(args.initial_checkpoint)
    booster.load_state_dict(checkpoint['feature_booster'])

if args.use_validation:
    validation_dataset.build_dataset()
    min_validation_loss = process_epoch(
        validation_dataloader, booster, optimizer, start_epoch - 1, train=False
    )

# Start the training
# training_dataset.build_dataset()
for epoch_idx in range(start_epoch, args.num_epochs + 1):

    # Process epoch
    training_dataset.build_dataset()
    train_loss_history.append(
        process_epoch(training_dataloader, booster, optimizer, epoch_idx)
    )

    if args.use_validation:
        validation_loss_history.append(
            process_epoch(validation_dataloader, booster, optimizer, epoch_idx, train=False)
        )

    # Save the current checkpoint
    checkpoint_path = os.path.join(result_directory, 'checkpoints', 'epoch%02d.pth' % epoch_idx)
    checkpoint = {
        'arg': args,
        'epoch_idx': epoch_idx,
        'feature_booster': booster.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history
    }
    torch.save(checkpoint, checkpoint_path)

    if (args.use_validation and validation_loss_history[-1] < min_validation_loss):
        min_validation_loss = validation_loss_history[-1]
        best_checkpoint_path = os.path.join(result_directory, 'best.pth')
        shutil.copy(checkpoint_path, best_checkpoint_path)

# Close the log file
log_file.close()