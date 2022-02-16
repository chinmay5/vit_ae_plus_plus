import argparse
import os

from tqdm import tqdm

from dataset.dataset_factory import get_dataset
from read_configs import bootstrap

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed


def plot_img_util(val_input):
    scaled_input = (val_input - val_input.min()) / (val_input.max() - val_input.min())  # [-1, 1]
    scaled_input = (scaled_input + 1) / 2  # [0, 1]
    scaled_input = (255 * scaled_input).to(torch.uint8)
    return scaled_input


def process_und_generate_mask(model, mask):
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)  # (N, L*H*W, p*p*1) 1=num_channel
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nclhw->nlhwc', mask).detach().cpu()
    mask = mask.detach()
    return mask


@torch.no_grad()
def check_reconstruction(data_loader, model, device, log_writer=None):
    # switch to evaluation mode
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    input_img = torch.FloatTensor().to(device)
    maskTensor = torch.FloatTensor().to(device)

    for batch in tqdm(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            _, output, mask = model(images)
            output = model.unpatchify(output)
            print(f"\nThe pred fraction is {(output > 0).sum() / (output.size(0) * 96 * 96 * 96)}")
            print(f"\nThe gt fraction is {(images > 0).sum() / (images.size(0) * 96 * 96 * 96)}")
            mask = process_und_generate_mask(model=model, mask=mask)
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)
        input_img = torch.cat((input_img, images), 0)
        maskTensor = torch.cat((maskTensor, mask), 0)

    if log_writer is not None:
        gt_img = plot_img_util(input_img[0].squeeze_().unsqueeze(1))
        output = plot_img_util(outPRED[0].squeeze_().unsqueeze(1))
        mask_img = plot_img_util(maskTensor[0].squeeze_().unsqueeze(1))
        log_writer.add_images(tag='feat_set', img_tensor=output)
        log_writer.add_images(tag='input_img', img_tensor=gt_img)
        log_writer.add_images(tag='mask_img', img_tensor=mask_img)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE ssl feature extraction module', add_help=False)
    # parser.add_argument('--batch_size', default=4, type=int,
    #                     help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--dataset', default='brats', type=str,
    #                     help='dataset name')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')

    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')

    # parser.add_argument('--patch_size', default=8, type=int,
    #                     help='Patch size for dividing the input')

    # * Finetuning params
    parser.add_argument('--global_pool', action='store_true')
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    # parser.add_argument('--log_dir', default='output_dir/logs',
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args = bootstrap(args=args, key='SANITY')

    dataset_train = get_dataset(dataset_name=args.dataset, mode='train', args=args, use_z_score=args.use_z_score)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, use_z_score=args.use_z_score)

    # Create the directory for saving the features
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir')
    os.makedirs(ssl_feature_dir, exist_ok=True)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = get_models(model_name='autoenc', args=args)
    args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
    test_writer = SummaryWriter(args.log_dir)

    args.finetune = os.path.join(args.output_dir, "checkpoints", args.checkpoint)

    if args.finetune and not args.eval:
        args.finetune = os.path.join(PROJECT_ROOT_DIR, args.finetune)
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
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
    check_reconstruction(data_loader_train, model, device, log_writer=None)
    check_reconstruction(data_loader_test, model, device, log_writer=test_writer)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
