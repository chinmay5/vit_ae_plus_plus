import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import pickle
import sys
import time
import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from timm.data import Mixup

from post_training_utils.fine_tune_epoch import train_one_epoch, evaluate_best_val_model, select_best_model, evaluate
from utils.custom_loss import SoftCrossEntropyWithWeightsLoss
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from k_fold_training_scripts.train_3d_resnet import get_all_feat_and_labels
from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap
from utils import misc
import torchio as tio


class MixUp3D:
    def __init__(self, mixup_alpha):
        super(MixUp3D, self).__init__()
        self.mixup_alpha = mixup_alpha

    def partial_mixup(self, input, indices):
        if input.size(0) != indices.size(0):
            raise RuntimeError("Size mismatch!")
        perm_input = input[indices]
        lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return input.mul(lam_mix).add(perm_input, alpha=1 - lam_mix)

    def __call__(self, input, target):
        indices = torch.randperm(input.size(0), device=input.device, dtype=torch.long)
        return self.partial_mixup(input, indices), self.partial_mixup(target, indices)


def get_args_parser():
    parser = argparse.ArgumentParser('K-fold cross validation', add_help=False)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    # Model parameters
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
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

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
    parser.add_argument('--use_mixup', action='store_true')

    return parser


def mean(input_arr):
    return sum(input_arr) / len(input_arr)


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
    args = bootstrap(args=args, key='FINE_TUNE_K_FOLD')
    train_transforms = tio.Compose(transforms)
    print(f"Masking ratio is {args.mask_ratio}")
    dataset_whole = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=train_transforms,
                                use_z_score=args.use_z_score)
    # Since we do not want any augmentation on the test set.
    dataset_whole_no_aug = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=None,
                                       use_z_score=args.use_z_score)
    features, labels = get_all_feat_and_labels(dataset_whole, args=args)
    # Code for the K-fold cross validation
    log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    kfold_splits = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
    # Create the location for storing splits
    split_index_path = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'k_fold', 'indices_file')
    os.makedirs(split_index_path, exist_ok=True)
    # Computing roc-auc for best models based on different criterions
    test_ft_best, test_spec_best, test_sens_best = [], [], []
    for idx, (train_ids, test_ids) in enumerate(kfold_splits.split(features, labels)):
        # First we need to save these indices. This would ensure we can reproduce the results
        print(f"Starting for fold {idx}")
        if os.path.exists(os.path.join(split_index_path, f"train_{idx}")) and \
                os.path.exists(os.path.join(split_index_path, f"test_{idx}")):
            train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{idx}"), 'rb'))
            test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))
        else:
            print("WARNING: Creating fresh splits.")
            pickle.dump(train_ids, open(os.path.join(split_index_path, f"train_{idx}"), 'wb'))
            pickle.dump(test_ids, open(os.path.join(split_index_path, f"test_{idx}"), 'wb'))
        # Let us generate the validation split
        train_ids, val_ids, _, _ = train_test_split(
            train_ids, labels[train_ids], test_size=0.20, random_state=42)
        # A quick sanity check
        assert set(train_ids).isdisjoint(set(test_ids)) == set(train_ids).isdisjoint(set(val_ids)) == set(
            val_ids).isdisjoint(set(test_ids)) == True, "Something went wrong"
        # # Now we create the dataloader
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_whole, sampler=train_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_whole_no_aug, sampler=val_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        # Since we want no augmentation on the test set
        data_loader_test = torch.utils.data.DataLoader(
            dataset_whole_no_aug, sampler=test_subsampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        # We get the vit autoencoder model
        model = get_models(model_name='vit', args=args)

        if args.dataset == 'brats':
            print("Using 3, 1 weight for Brats")
            args.cross_entropy_wt = torch.as_tensor([3.0, 1.0])

        if args.eval:
            # Load the weights of the best model
            load_weights_and_evaluate(args, idx, model, 'spec', data_loader_test, test_subsampler, device)
            load_weights_and_evaluate(args, idx, model, 'sens', data_loader_test, test_subsampler, device)
            test_ft_best.append(load_weights_and_evaluate(args, idx, model, 'ft', data_loader_test, test_subsampler, device))

            if idx == 2:
                print(f"Final result is {mean(test_ft_best)}")
                sys.exit(0)
            else:
                continue

        # The log writers
        args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer_train = SummaryWriter(log_dir=f"{args.log_dir}/train_ft")
        log_writer_val = SummaryWriter(log_dir=f"{args.log_dir}/val_ft")
        args.finetune = os.path.join(PROJECT_ROOT_DIR, args.output_dir, "checkpoints",
                                     f'checkpoint-min_loss_k_fold_split_{idx}.pth')
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

        torch.nn.init.trunc_normal_(model.head.weight, std=2e-5)

        if args.fix_backbone:
            print("fixing the backbone")
            # Fix the feature extraction layers
            for name, param in model.named_parameters():
                if name not in ['head.weight', 'head.bias']:
                    param.requires_grad = False

        model.to(device)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # print("Model = %s" % str(model_without_ddp))
        print('number of params (M): %.2f' % (n_parameters / 1.e6))

        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

        if args.lr is None:  # only base_lr is specified
            args.lr = args.blr * eff_batch_size / 256

        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)

        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module

        # build optimizer with layer-wise lr decay (lrd)
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias

        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
        loss_scaler = NativeScaler()


        mixup_fn = None
        if args.use_mixup:
            # smoothing is handled with mixup label transform
            mixup_fn = MixUp3D(mixup_alpha=0.1)
            criterion = SoftCrossEntropyWithWeightsLoss(weights=args.cross_entropy_wt).to(device)
        else:
            print("Default case criterion")
            criterion = torch.nn.CrossEntropyLoss(weight=args.cross_entropy_wt).to(device)

        print("criterion = %s" % str(criterion))
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        print(f"Start training for {args.epochs} epochs")

        start_time = time.time()
        max_roc_auc_score, max_spec, max_sen = 0.0, 0.0, 0.0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mix_up_fn=mixup_fn,
                log_writer=log_writer_train,
                args=args
            )
            # Let us record both train and val stats
            train_val_stats = evaluate(data_loader_train, model, device, args=args)
            val_stats = evaluate(data_loader_val, model, device, args=args)

            print(
                f"ROC_AUC score of the network on the {len(val_subsampler)} val images: {val_stats['roc_auc_score']:.1f}%")
            max_roc_auc_score = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler,
                                                  max_val=max_roc_auc_score,
                                                  model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                                  cur_val=val_stats['roc_auc_score'],
                                                  model_name=f'best_ft_model_{idx}')
            # Let us save model based on the other criterions
            max_spec = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler, max_val=max_spec,
                                         model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                         cur_val=val_stats['specificity'], model_name=f'best_spec_model_{idx}')
            max_sen = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler, max_val=max_sen,
                                        model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                        cur_val=val_stats['sensitivity'], model_name=f'best_sens_model_{idx}')

            # Writing the logs
            log_writer_val.add_scalar('ft/roc_auc_score', val_stats['roc_auc_score'], epoch)
            log_writer_val.add_scalar('ft/loss', val_stats['loss'], epoch)
            log_writer_train.add_scalar('ft/roc_auc_score', train_val_stats['roc_auc_score'], epoch)
            log_writer_train.add_scalar('ft/loss', train_val_stats['loss'], epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'train_val_{k}': v for k, v in train_val_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if misc.is_main_process():
                log_writer_train.flush()
                log_writer_val.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        # Now, let us load the best model and evaluate
        # For the dataset_test parameter, we just need the len() function to be defined. Hence, it will work here.
        test_ft_best.append(
            evaluate_best_val_model(args=args, data_loader_test=data_loader_test, dataset_test=test_subsampler,
                                    device=device, model=model, model_name=f'best_ft_model_{idx}')
        )
        test_spec_best.append(
            evaluate_best_val_model(args=args, data_loader_test=data_loader_test, dataset_test=test_subsampler,
                                    device=device, model=model, model_name=f'best_spec_model_{idx}')
        )
        test_sens_best.append(
            evaluate_best_val_model(args=args, data_loader_test=data_loader_test, dataset_test=test_subsampler,
                                    device=device, model=model, model_name=f'best_sens_model_{idx}')
        )
        del model
        torch.cuda.empty_cache()
    print("Result values")
    print(f"Best FT model -> {test_ft_best}, with mean value {mean(test_ft_best)}")
    print(f"Best Spec model -> {test_spec_best}, with mean value {mean(test_spec_best)}")
    print(f"Best Sens model -> {test_sens_best}, with mean value {mean(test_sens_best)}")


def load_weights_and_evaluate(args, idx, model, model_metric, data_loader_test, test_subsampler, device):
    args.finetune = os.path.join(PROJECT_ROOT_DIR, args.output_dir, f'checkpoint-best_{model_metric}_model_{idx}.pth')
    checkpoint = torch.load(args.finetune, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=False)
    assert len(msg.missing_keys) == 0, "All keys did not match"
    # Since we just need to define the `len` function in this case.
    roc = evaluate_best_test_model(args=args, data_loader_test=data_loader_test, dataset_test=test_subsampler,
                             device=device, model=model, model_name=f'best_{model_metric}')
    del model
    torch.cuda.empty_cache()
    return roc


def evaluate_best_test_model(args, data_loader_test, dataset_test, device, model, model_name):
    model.to(device)
    test_stats = evaluate(data_loader=data_loader_test, model=model, device=device, args=args)
    print(f"Accuracy of {model_name} on the {len(dataset_test)} test images: {test_stats['roc_auc_score']:.1f}%")
    return test_stats['roc_auc_score']


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
