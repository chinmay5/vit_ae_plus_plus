import argparse
import math
import os
import pickle

from sklearn.model_selection import StratifiedKFold

from ablation.resnet_3d import generate_model
from dataset.dataset_factory import get_dataset
from read_configs import bootstrap
from utils.used_metrics import roc_auc

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

import numpy as np
import torch
from torch.backends import cudnn

from utils import misc, lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from torch.utils.tensorboard import SummaryWriter
from environment_setup import PROJECT_ROOT_DIR
import torchio as tio


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, args=None, mix_up_fn=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mix_up_fn is not None:
            samples, targets = mix_up_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            targets = targets.to(torch.long)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    # Weights for breast_tumor = 2:1 majority being label 0
    # Since evaluation is always hard target and not SoftTarget
    criterion = torch.nn.CrossEntropyLoss().to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)
        metric_logger.update(loss=loss.item())

    roc_score, spec, sens = roc_auc(predictions=outPRED, target=outGT)
    metric_logger.update(roc_score=roc_score)
    metric_logger.update(spec=spec)
    metric_logger.update(sens=sens)
    metric_logger.synchronize_between_processes()
    print('* acc {:.3f}, loss {losses.global_avg:.3f}'
          .format(roc_score, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--patch_size', default=8, type=int,
                        help='Patch size for dividing the input')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_all_feat_und_labels(dataset_whole, args):
    mri, labels = [], []
    for idx in range(len(dataset_whole)):
        mri.append(dataset_whole[idx][0])
        labels.append(dataset_whole[idx][-1])
    if args.in_channels == 1:
        return torch.cat(mri), torch.stack(labels)
    else:
        return torch.stack(mri), torch.stack(labels)


def main(args, train=True):
    device = torch.device(args.device)
    if train:
        misc.init_distributed_mode(args)

        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

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
        args = bootstrap(args=args, key='RESNET')
        train_transforms = tio.Compose(transforms)
        dataset_whole = get_dataset(dataset_name=args.dataset, mode='whole', args=args, transforms=train_transforms,
                                    use_z_score=args.use_z_score)
        features, labels = get_all_feat_und_labels(dataset_whole, args=args)
        max_roc_k_fold = 0
        # Code for the K-fold cross validation
        args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, 'checkpoints')
        os.makedirs(args.output_dir, exist_ok=True)
        kfold_splits = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        for idx, (train_ids, test_ids) in enumerate(kfold_splits.split(features, labels)):
            pickle.dump(train_ids, open(os.path.join(args.output_dir, f"train_{idx}"), 'wb'))
            pickle.dump(test_ids, open(os.path.join(args.output_dir, f"test_{idx}"), 'wb'))
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            data_loader_train = torch.utils.data.DataLoader(
                dataset_whole, sampler=train_subsampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )

            data_loader_test = torch.utils.data.DataLoader(
                dataset_whole, sampler=test_subsampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            model = generate_model(model_depth=10, n_classes=args.num_classes, n_input_channels=args.in_channels)

            # Initialize optimizer
            args.lr = 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            # Loss function
            print("-------------------------------NOTE---------------------------------------")
            print("Using 3, 1 weight for Brats")
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([3.0, 1.0])).to(device)

            model.to(device)

            model_without_ddp = model
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print("Model = %s" % str(model_without_ddp))
            print('number of params (M): %.2f' % (n_parameters / 1.e6))
            loss_scaler = NativeScaler()

            # Create the summary writers
            args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer_train = SummaryWriter(log_dir=f"{args.log_dir}/train_ft_{idx}")
            log_writer_val = SummaryWriter(log_dir=f"{args.log_dir}/val_ft_{idx}")
            # Run the training loop for defined number of epochs
            roc_score = 0
            for epoch in range(0, args.epochs):
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, mix_up_fn=None,
                    log_writer=log_writer_train,
                    args=args
                )
                # Let us record both train and val stats
                train_val_stats = evaluate(data_loader_train, model, device, args=args)
                test_stats = evaluate(data_loader_test, model, device, args=args)

                print(
                    f"Accuracy of the network on the {len(test_stats)} val images: {test_stats['roc_score']:.1f}%")
                roc_score = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler,
                                            max_val=roc_score,
                                            model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                            cur_val=test_stats['roc_score'],
                                            model_name=f'best_ft_model_split{idx}')
                # Writing the logs
                log_writer_val.add_scalar('ft/acc', test_stats['roc_score'], epoch)
                log_writer_val.add_scalar('ft/loss', test_stats['loss'], epoch)
                log_writer_train.add_scalar('ft/roc_auc_score', train_val_stats['roc_score'], epoch)
                log_writer_train.add_scalar('ft/loss', train_val_stats['loss'], epoch)

                # Process is complete.
                print(f'Epoch - {epoch} complete')

            print(f'Final roc value is {max_roc_k_fold / 5}')
    else:
        # We are in the evaluation mode
        print("Using Test-Only Mode")
        args = bootstrap(args=args, key='RESNET')
        dataset_whole = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                                    use_z_score=args.use_z_score)
        # features, labels = get_all_feat_und_labels(dataset_whole, args=args)
        max_roc_k_fold = 0
        # Code for the K-fold cross validation
        args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir, 'checkpoints')
        os.makedirs(args.output_dir, exist_ok=True)
        # kfold_splits = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        avg_spec, avg_sen, avg_roc_score = 0, 0, 0
        for idx in range(3):
            train_ids = pickle.load(open(os.path.join(args.output_dir, f"train_{idx}"), 'rb'))
            test_ids = pickle.load(open(os.path.join(args.output_dir, f"test_{idx}"), 'rb'))
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            data_loader_test = torch.utils.data.DataLoader(
                dataset_whole, sampler=test_subsampler,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

            model = generate_model(model_depth=10, n_classes=args.num_classes, n_input_channels=args.in_channels)
            args.finetune = os.path.join(args.output_dir, f'checkpoint-best_ft_model_split{idx}.pth')
            checkpoint = torch.load(args.finetune, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % args.finetune)
            checkpoint_model = checkpoint['model']
            model.load_state_dict(checkpoint_model)
            model.to(device)
            test_stats = evaluate(data_loader_test, model, device, args=args)
            avg_sen += test_stats['sens']
            avg_spec += test_stats['spec']
            avg_roc_score += test_stats['roc_score']
        print(f"Specificity {avg_spec/3} and Sensitivity {avg_sen/3}")



def select_best_model(args, epoch, loss_scaler, max_val, model, model_without_ddp, optimizer, cur_val,
                      model_name='best_ft_model'):
    if cur_val > max_val:
        print(f"saving {model_name} @ epoch {epoch}")
        max_val = cur_val
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=model_name)  # A little hack for saving model with preferred name
    return max_val


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, train=True)
