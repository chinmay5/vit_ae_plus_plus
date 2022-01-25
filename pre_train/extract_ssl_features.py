import argparse
import os

from tqdm import tqdm

from dataset.dataset_factory import get_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from dataset.brain_tumor.pretrain_tumor_data import build_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed



@torch.no_grad()
def generate_features(data_loader, model, device, ssl_feature_dir, feature_file_name='features.npy', label_file_name='gt_labels.npy', log_writer=None):
    # switch to evaluation mode
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for batch in tqdm(data_loader):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward_features(images)
        outPRED = torch.cat((outPRED, output), 0)
        outGT = torch.cat((outGT, target), 0)
    if feature_file_name is not None:
        print("Saving features!!!")
        np.save(os.path.join(ssl_feature_dir, feature_file_name), outPRED.cpu().numpy())
    if label_file_name is not None:
        print("Saving labels!!!")
        np.save(os.path.join(ssl_feature_dir, label_file_name), outGT.cpu().numpy())
    if log_writer is not None:
        metadata = [x.item() for x in outGT]
        log_writer.add_embedding(outPRED, metadata=metadata, tag='ssl_embedding')


def get_args_parser():
    parser = argparse.ArgumentParser('MAE ssl feature extraction module', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--dataset', default='brats', type=str,
                        help='dataset name')
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')

    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')

    parser.add_argument('--patch_size', default=8, type=int,
                        help='Patch size for dividing the input')

    # * Finetuning params
    parser.add_argument('--finetune', default='output_dir/checkpoints/checkpoint-240.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--log_dir', default='output_dir/finetune_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

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


    dataset_train = get_dataset(dataset_name=args.dataset, mode='feat_extract', args=args, use_z_score=True)
    dataset_test = get_dataset(dataset_name=args.dataset, mode='test', args=args, use_z_score=True)

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

    model = get_models(model_name='vit', args=args)
    args.log_dir = os.path.join(PROJECT_ROOT_DIR, args.log_dir)
    train_writer = SummaryWriter(args.log_dir)

    if args.finetune and not args.eval:
        args.finetune = os.path.join(PROJECT_ROOT_DIR, args.finetune)
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
    generate_features(data_loader_train, model, device, log_writer=train_writer, ssl_feature_dir=ssl_feature_dir)
    generate_features(data_loader_test, model, device, feature_file_name='test_ssl_features.npy', label_file_name=None, ssl_feature_dir=ssl_feature_dir)
    # Also, let us save the vit model. We need not go through the entire process of getting the vit from autoenc everytime
    ssl_file_name = os.path.join(PROJECT_ROOT_DIR, 'output_dir', 'checkpoints', 'ssl_feat.pth')
    torch.save(model.state_dict(), ssl_file_name)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
