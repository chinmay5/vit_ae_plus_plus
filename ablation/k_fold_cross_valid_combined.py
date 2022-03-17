import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import pickle
import time
import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from timm.optim import optim_factory

import fine_tune
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import pre_train
from ablation.train_3d_resnet import get_all_feat_und_labels
from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap
from utils import misc
import torchio as tio


from utils.feature_extraction import generate_features


def get_args_parser():
    parser = argparse.ArgumentParser('K-fold cross validation', add_help=False)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
    parser.add_argument('--model', default='contr_mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0, metavar='LR',  # earlier 0
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--dist_on_itp', action='store_true')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # device = torch.device(args.device)
    device = torch.device("cuda:0")

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transforms = [
        tio.RandomAffine(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    args = bootstrap(args=args, key='K_FOLD')
    train_transforms = tio.Compose(transforms)
    print(f"Masking ratio is {args.mask_ratio}")
    dataset_whole = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=train_transforms,
                                use_z_score=args.use_z_score)
    # Since we do not want any augmentation on the test set.
    dataset_whole_no_aug = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=None,
                                       use_z_score=args.use_z_score)
    features, labels = get_all_feat_und_labels(dataset_whole, args=args)
    # Code for the K-fold cross validation
    log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir)
    kfold_splits = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
    # Create the location for storing splits
    split_index_path = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'k_fold', 'indices_file')
    os.makedirs(split_index_path, exist_ok=True)
    for idx, (train_ids, test_ids) in enumerate(kfold_splits.split(features, labels)):
        # First we need to save these indices. This would ensure we can reproduce the results
        print(f"Starting for fold {idx}")
        # pickle.dump(train_ids, open(os.path.join(split_index_path, f"train_{idx}"), 'wb'))
        # pickle.dump(test_ids, open(os.path.join(split_index_path, f"test_{idx}"), 'wb'))

        train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{idx}"), 'rb'))
        test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))

        # Needed for the pre-training phase
        args.nb_classes = 2
        #
        # # Now we create the dataloader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        #
        data_loader_train = torch.utils.data.DataLoader(
            dataset_whole, sampler=train_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        # Since we want no augmentation on the test set
        data_loader_test = torch.utils.data.DataLoader(
            dataset_whole_no_aug, sampler=test_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        # We first start with the training of our vit_autoenc model
        model = get_models(model_name='autoenc', args=args)

        model.to(device)

        model_without_ddp = model
        print("Model = %s" % str(model_without_ddp))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        print(optimizer)
        loss_scaler = NativeScaler()

        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, 'checkpoints')
        os.makedirs(args.output_dir, exist_ok=True)

        print(f"Start training for {args.epochs} epochs")

        start_time = time.time()
        min_loss = float('inf')
        for epoch in range(args.start_epoch, args.epochs):
            # loss weighting for the edge maps
            if args.use_edge_map:
                edge_map_weight = 0
            else:
                edge_map_weight = 0.01 * (1 - epoch / args.epochs)
            train_stats = pre_train.train_one_stage_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args,
                edge_map_weight=edge_map_weight
            )

            if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
            if train_stats['loss'] < min_loss:
                min_loss = train_stats['loss']
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=f"min_loss_k_fold_split_{idx}")

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # Now we would go ahead and also do the feature extraction for this split
        del model
        torch.cuda.empty_cache()
        # Starting the extraction process
        model = get_models(model_name='vit', args=args)
        args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)

        args.finetune = os.path.join(args.output_dir, f"checkpoint-min_loss_k_fold_split_{idx}.pth")
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        model.to(device)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir', args.subtype)
        os.makedirs(ssl_feature_dir, exist_ok=True)
        print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))
        # To have deterministic elements, we are going to get the dataset again with the correct indices.
        # The indices should be placed sequentially else we run into issues once we try the combination
        dataset_no_aug_train = torch.utils.data.Subset(dataset_whole_no_aug, train_ids)
        dataset_no_aug_test = torch.utils.data.Subset(dataset_whole_no_aug, test_ids)

        data_loader_train_feat_extr = torch.utils.data.DataLoader(
            dataset_no_aug_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        # Since we want no augmentation on the test set
        data_loader_test_feat_extr = torch.utils.data.DataLoader(
            dataset_no_aug_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        generate_features(data_loader_train_feat_extr, model, device,
                          feature_file_name=f'train_contrast_ssl_features_split_{idx}.npy',
                          label_file_name=f'train_contrast_ssl_labels_split_{idx}.npy',
                          ssl_feature_dir=ssl_feature_dir)
        generate_features(data_loader_test_feat_extr, model, device,
                          feature_file_name=f'test_contrast_ssl_features_split_{idx}.npy',
                          label_file_name=f'test_contrast_ssl_labels_split_{idx}.npy',
                          ssl_feature_dir=ssl_feature_dir)
        #################################################################
        ###### Contrastive Training
        #################################################################
        # del model
        # torch.cuda.empty_cache()
        # print("Now we start with the contrastive training part")
        # # A required change for using contrastive training. We want to match the feature similarity hence, obtain the features
        # # directly rather than projecting them first
        # args.nb_classes = -1
        #
        # cointrast_log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, 'contrast')
        # log_writer_contrast = SummaryWriter(log_dir=cointrast_log_dir)
        #
        # model = get_models(model_name='contrastive', args=args)
        #
        # args.finetune = os.path.join(args.output_dir, f"checkpoint-min_loss_k_fold_split_{idx}.pth")
        # checkpoint = torch.load(args.finetune, map_location='cpu')
        #
        # print("Load pre-trained checkpoint from: %s" % args.finetune)
        # checkpoint_model = checkpoint['model']
        # state_dict = model.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        #
        # # interpolate position embedding
        # interpolate_pos_embed(model, checkpoint_model)
        #
        # # load pre-trained model
        # msg = model.load_state_dict(checkpoint_model, strict=False)
        # print(msg)
        #
        # model.to(device)
        #
        # model_without_ddp = model
        # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #
        # print("Model = %s" % str(model_without_ddp))
        # print('number of params (M): %.2f' % (n_parameters / 1.e6))
        #
        # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        #
        # if args.lr is None:  # only base_lr is specified
        #     args.lr = args.blr * eff_batch_size / 256
        #
        # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        # print("actual lr: %.2e" % args.lr)
        #
        # print("accumulate grad iterations: %d" % args.accum_iter)
        # print("effective batch size: %d" % eff_batch_size)
        #
        # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
        # optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
        # loss_scaler = NativeScaler()
        #
        # criterion = torch.nn.CosineSimilarity(dim=1).to(device)
        #
        # print("criterion = %s" % str(criterion))
        #
        # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        #
        # print(f"Start training for {args.epochs} epochs")
        # min_loss = float("inf")
        # start_time = time.time()
        # for epoch in range(args.start_epoch, args.epochs):
        #     train_stats = fine_tune.contrastive_training.train_one_epoch(
        #         model, criterion, data_loader_train,
        #         optimizer, device, epoch, loss_scaler,
        #         args.clip_grad, log_writer=log_writer_contrast,
        #         args=args
        #     )
        #     if train_stats['loss'] < min_loss:
        #         min_loss = train_stats['loss']
        #         misc.save_model(
        #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #             loss_scaler=loss_scaler,
        #             epoch=f"contrastive_model_k_fold_split_{idx}")  # A little hack for saving model with preferred name
        #     if misc.is_main_process():
        #         log_writer_contrast.flush()
        # # Add a logic for saving the trained model
        #
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Training time {}'.format(total_time_str))
        #
        # # Now let's extract the features
        #
        # del model
        # torch.cuda.empty_cache()
        # model = get_models(model_name='contrastive', args=args)
        #
        # model_path = os.path.join(args.output_dir, f'checkpoint-contrastive_model_k_fold_split_{idx}.pth')
        # print(model_path)
        # assert os.path.exists(model_path), "Please ensure a trained model alredy exists"
        # checkpoint = torch.load(model_path, map_location='cpu')
        # model.load_state_dict(checkpoint['model'])
        # model.to(device)
        #
        # contrastive_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir', args.subtype)
        # os.makedirs(contrastive_feature_dir, exist_ok=True)
        #
        # generate_features(data_loader_train_feat_extr, model, device,
        #                   feature_file_name=f'train_contrast_ssl_features_split_{idx}.npy',
        #                   label_file_name=f'train_contrast_ssl_labels_split_{idx}.npy',
        #                   ssl_feature_dir=ssl_feature_dir)
        # generate_features(data_loader_test_feat_extr, model, device,
        #                   feature_file_name=f'test_contrast_ssl_features_split_{idx}.npy',
        #                   label_file_name=f'test_contrast_ssl_labels_split_{idx}.npy',
        #                   ssl_feature_dir=ssl_feature_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
