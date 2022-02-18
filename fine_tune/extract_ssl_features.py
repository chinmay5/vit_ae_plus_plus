import argparse
import os

from dataset.dataset_factory import get_dataset
from read_configs import bootstrap
from utils.feature_extraction import generate_features

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models
from model.model_utils.vit_helpers import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser('MAE ssl feature extraction module', add_help=False)

    parser.add_argument('--volume_size', default=96, type=int,
                        help='images input size')

    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of channels in the input')

    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=32, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')

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

    args = bootstrap(args=args, key='EXTRACT_SSL')
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

    # Create the directory for saving the features
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, args.dataset, 'ssl_features_dir', args.subtype)
    os.makedirs(ssl_feature_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
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

    if args.global_pool:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    else:
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    if not args.only_test_split:
        generate_features(data_loader_train, model, device, feature_file_name='train_ssl_features.npy',
                          label_file_name='train_ssl_labels.npy',
                          ssl_feature_dir=ssl_feature_dir)
    generate_features(data_loader_test, model, device, feature_file_name='test_ssl_features.npy',
                      label_file_name='test_ssl_labels.npy',
                      ssl_feature_dir=ssl_feature_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
