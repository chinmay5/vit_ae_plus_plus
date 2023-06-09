import argparse
import math
import os

from dataset.dataset_factory import get_dataset
from read_configs import bootstrap
from utils.used_metrics import roc_auc

import sys

import numpy as np
import torch
from torch.backends import cudnn

from model.model_utils.vit_helpers import interpolate_pos_embed

from model.model_factory import get_models

from utils import misc, lr_sched
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

from torch.utils.tensorboard import SummaryWriter
from environment_setup import PROJECT_ROOT_DIR
import utils.lr_decay as lrd
import time
import json
import datetime
import torchio as tio
from utils.custom_loss import SoftCrossEntropyWithWeightsLoss

from timm.data.mixup import Mixup


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
    criterion = torch.nn.CrossEntropyLoss(weight=args.cross_entropy_wt).to(device)

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

        # batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['specificity'].update(specificity, n=batch_size)
        # metric_logger.meters['sensitivity'].update(sensitivity, n=batch_size)
    roc_auc_score, specificity, sensitivity = roc_auc(predictions=outPRED, target=outGT)
    metric_logger.update(roc_auc_score=roc_auc_score)
    metric_logger.update(specificity=specificity)
    metric_logger.update(sensitivity=sensitivity)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* roc_auc_score {:.3f}, loss {losses.global_avg:.3f}'
          .format(roc_auc_score, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    # parser.add_argument('--batch_size', default=4, type=int,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')

    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--patch_size', default=8, type=int,
                        help='Patch size for dividing the input')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--weight_decay', type=float, default=0.05,
    #                     help='weight decay (default: 0.05)')

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


    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')


    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')


    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true',
    #                     help='Perform evaluation only')
    # parser.set_defaults(eval=True)
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


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    transforms = [
        tio.RandomAffine(),
        # tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    args = bootstrap(args=args, key='FINE_TUNE')
    train_transforms = tio.Compose(transforms)
    dataset_train = get_dataset(dataset_name=args.dataset, mode='train', args=args, transforms=train_transforms,
                                use_z_score=args.use_z_score)
    dataset_val = get_dataset(dataset_name=args.dataset, mode='valid', args=args, transforms=None,
                              use_z_score=args.use_z_score)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                               use_z_score=args.use_z_score)


    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if args.log_dir is not None and not args.eval:
        args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer_train = SummaryWriter(log_dir=f"{args.log_dir}/train_ft")
        log_writer_val = SummaryWriter(log_dir=f"{args.log_dir}/val_ft")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = get_models(model_name='vit', args=args)

    # Align the generated folders with our structure
    args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_ft_model',
                                mode='test')
        evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_spec_model',
                                mode='test')
        evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_sens_model',
                                mode='test')
        exit(0)

    if not args.eval:
        args.finetune = os.path.join(PROJECT_ROOT_DIR, args.feature_extractor_load_path, "checkpoints", args.checkpoint)
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

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
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
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    mixup_fn = None
    if args.use_mixup:
        # smoothing is handled with mixup label transform
        mixup_fn = Mixup(mixup_alpha=0.1, num_classes=2)
        criterion = SoftCrossEntropyWithWeightsLoss(weights=args.cross_entropy_wt).to(device)
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    # criterion = torch.nn.CrossEntropyLoss()  # Label 1 occurs ~3 times more than 0
    else:
        print("Default case criterion")
        criterion = torch.nn.CrossEntropyLoss(weight=args.cross_entropy_wt).to(device)
    # criterion = torch.nn.BCEWithLogitsLoss()list_val = [dataset_train[idx][1].item() for idx in range(len(dataset_tra

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

        print(f"ROC_AUC score of the network on the {len(dataset_val)} val images: {val_stats['roc_auc_score']:.1f}%")
        max_roc_auc_score = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler,
                                              max_val=max_roc_auc_score,
                                              model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                              cur_val=val_stats['roc_auc_score'],
                                              model_name='best_ft_model')
        # Let us save model based on the other criterions
        max_spec = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler, max_val=max_spec,
                                     model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                     cur_val=val_stats['specificity'], model_name='best_spec_model')
        max_sen = select_best_model(args=args, epoch=epoch, loss_scaler=loss_scaler, max_val=max_sen,
                                    model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                    cur_val=val_stats['sensitivity'], model_name='best_sens_model')

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
    evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_ft_model')
    evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_spec_model')
    evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_sens_model')


def evaluate_best_val_model(args, data_loader_test, dataset_test, device, model, model_name='best_ft_model.pth',
                            mode=None):
    if mode == 'test':
        checkpoint = torch.load(os.path.join(PROJECT_ROOT_DIR, args.eval_model_path, f'checkpoint-{model_name}.pth'),
                                map_location='cpu')
    else:
        checkpoint = torch.load(os.path.join(args.output_dir, f'checkpoint-{model_name}.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    test_stats = evaluate(data_loader=data_loader_test, model=model, device=device, args=args)
    print(f"Accuracy of {model_name} on the {len(dataset_test)} test images: {test_stats['roc_auc_score']:.1f}%")
    return test_stats['roc_auc_score']


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
    main(args)
