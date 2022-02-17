import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torchio as tio
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.lr_decay as lrd
from dataset.dataset_factory import get_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed
from read_configs import bootstrap
from utils import misc, lr_sched
from utils.feature_extraction import generate_features
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (augmented, original, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        augmented = augmented.to(device, non_blocking=True)
        original = original.to(device, non_blocking=True)
        # print(data_iter_step, accum_iter)
        with torch.cuda.amp.autocast():
            p1, p2, z1, z2 = model(original, augmented)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

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


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
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

    # Dataset parameters
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
    args = bootstrap(args=args, key='CONTRASTIVE')
    train_transforms = tio.Compose(transforms)
    dataset_train = get_dataset(dataset_name=args.dataset, mode='train', args=args, transforms=train_transforms,
                                use_z_score=args.use_z_score)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, transforms=None,
                               use_z_score=args.use_z_score)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.log_dir is not None:
        args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer_train = SummaryWriter(log_dir=f"{args.log_dir}/train_contrast")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = get_models(model_name='contrastive', args=args)

    # Align the generated folders with our structure
    args.output_dir = os.path.join(PROJECT_ROOT_DIR, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval:
        # we are going to extract features for test and train splits. We do not want any augmentation
        # on the training dataset
        dataset_train_no_aug = get_dataset(dataset_name=args.dataset, mode='train', args=args, transforms=None,
                                    use_z_score=args.use_z_score)
        data_loader_train_no_aug = torch.utils.data.DataLoader(
            dataset_train_no_aug,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        extract_features(args=args, data_loader_test=data_loader_test, data_loader_train=data_loader_train_no_aug,
                         device=device, model=model, log_writer_train=log_writer_train)
        exit(0)

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

    print("Cosine Similarity loss")
    criterion = torch.nn.CosineSimilarity(dim=1).to(device)

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    min_loss = float("inf")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer_train,
            args=args
        )
        if train_stats['loss'] < min_loss:
            min_loss = train_stats['loss']
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch="contrastive_model")  # A little hack for saving model with preferred name
        if misc.is_main_process():
            log_writer_train.flush()
    # Add a logic for saving the trained model

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def extract_features(args, data_loader_train, data_loader_test, device, model, log_writer_train):
    model_path = os.path.join(PROJECT_ROOT_DIR, args.eval_model_path, 'checkpoint-contrastive_model.pth')
    assert os.path.exists(model_path), "Please ensure a trained model alredy exists"
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    contrastive_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir', args.subtype)
    os.makedirs(contrastive_feature_dir, exist_ok=True)
    generate_features(data_loader_train, model, device, feature_file_name='train_contrast_ssl_features.npy',
                      label_file_name='train_contrast_ssl_labels.npy',
                      ssl_feature_dir=contrastive_feature_dir)
    generate_features(data_loader_test, model, device, feature_file_name='test_contrast_ssl_features.npy',
                      label_file_name='test_contrast_ssl_labels.npy',
                      ssl_feature_dir=contrastive_feature_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
