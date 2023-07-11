#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import argparse
import builtins
import numpy as np
import os
import random
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch import nn

from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from other_baselines.mocov3.moco.resent3d_base import generate_model
from other_baselines.mocov3.moco.vit_3d import VisionTransformer3D

model_names = ['vit_3d', 'resnet_3d']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# The dataset we are going to use
parser.add_argument('--use_z_score', type=bool, default=True)
parser.add_argument('--dataset', type=str, default='large_brats', help="dataset of our choice")
parser.add_argument("--mode", type=str, default='test',
                    help="The mode we have to use in the training")
parser.add_argument('--split', type=str, default='idh', help="split for the large brats dataset")

parser.add_argument('--in_channels', type=int, default=4)
parser.add_argument('--volume_size', type=int, default=96)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--only_test_split', action='store_true', help='Used for large brats when we have only the test split available')

parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_3d',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', type=str,
                    help='path to moco pretrained checkpoint',
                    default='vit_egd_data/min_loss.pth.tar')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet_3d':
        model = generate_model(model_depth=10, num_classes=args.num_classes, n_input_channels=args.in_channels)
        print("resnet3d model created")
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        linear_keyword = 'fc'
    else:
        # A negative num-classes ensures that the classifier layer is not created and rather, replaced with the Identity
        model = VisionTransformer3D(in_chans=args.in_channels, volume_size=args.volume_size, num_classes=-1)
        print("3d vision transformer model created")
        linear_keyword = 'head'

    print("Using ResNet feature backbone for feature extraction")
    for p in model.parameters():
        p.requires_grad = False

    # load from pre-trained, before DistributedDataParallel constructor
    if not args.pretrained:
        raise AttributeError("Please specify a pretrained model checkpoint.")
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        args.start_epoch = 0
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(args.pretrained))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    if args.only_test_split:
        print("Generating features for only the test split")
        dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                                   use_z_score=args.use_z_score)
        dataset_train = None
    else:
        dataset_train = get_dataset(dataset_name=args.dataset, mode='train', args=args, transforms=None,
                                    use_z_score=args.use_z_score)
        dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                                   use_z_score=args.use_z_score)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        train_sampler = None

    test_loader = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)

    extract_ssl_features(test_loader, model, args, feature_file_name='test_ssl_features.npy',
                         label_file_name='test_ssl_labels.npy', subfolder_name=args.arch)
    if dataset_train is not None:
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        extract_ssl_features(train_loader, model, args, feature_file_name='train_ssl_features.npy',
                             label_file_name='train_ssl_labels.npy', subfolder_name=args.arch)


def extract_ssl_features(val_loader, model, args, feature_file_name='features.npy', label_file_name='gt_labels.npy', subfolder_name='vit'):
    model.eval()
    outGT = torch.FloatTensor().cuda(args.gpu, non_blocking=True)
    outPRED = torch.FloatTensor().cuda(args.gpu, non_blocking=True)
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir', subfolder_name)
    os.makedirs(ssl_feature_dir, exist_ok=True)

    with torch.no_grad():
        for i, (images, _, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images).squeeze()
            outPRED = torch.cat((outPRED, output), 0)
            outGT = torch.cat((outGT, target), 0)
        if feature_file_name is not None:
            print("Saving features!!!")
            np.save(os.path.join(ssl_feature_dir, feature_file_name), outPRED.cpu().numpy())
        if label_file_name is not None:
            print("Saving labels!!!")
            np.save(os.path.join(ssl_feature_dir, label_file_name), outGT.cpu().numpy())


if __name__ == '__main__':
    main()
