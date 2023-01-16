import os
from configparser import ConfigParser

import torch

from environment_setup import PROJECT_ROOT_DIR


def read_config():
    """
    Function for reading and parsing configurations specific to the attack type
    :return: parsed configurations
    """
    config_path = os.path.join(PROJECT_ROOT_DIR, 'config.ini')
    parser = ConfigParser()
    parser.read(config_path)
    return parser


def read_float_lists(parser, key, variable):
    x = parser[key].get(variable)
    return [float(y) for y in x.strip().split("\n")]


def bootstrap(args, key='SETUP'):
    parser = read_config()
    args.dataset = parser['DATASET'].get('name')
    args.use_z_score = parser['DATASET'].getboolean('use_z_score')
    args.in_channels = parser['DATASET'].getint('in_channels', fallback=2)
    args.output_dir = parser[key].get('output_dir')
    args.log_dir = parser[key].get('log_dir')
    args.epochs = parser[key].getint('epochs')
    args.batch_size = parser[key].getint('batch_size')
    args.weight_decay = parser[key].getfloat('weight_decay')
    args.start_epoch = parser[key].getint('start_epoch')
    args.mask_ratio = parser[key].getfloat('mask_ratio')
    args.checkpoint = parser[key].get('checkpoint', fallback='checkpoint-380.pth')
    args.patch_size = parser[key].getint('patch_size', fallback=8)
    args.drop_path = parser[key].getfloat('drop_path', fallback=0)
    args.eval = parser[key].getboolean('eval', fallback=False)
    args.feature_extractor_load_path = parser[key].get('feature_extractor_load_path', fallback="")
    args.eval_model_path = parser[key].get('eval_model_path', fallback="")
    args.use_mixup = parser[key].getboolean('use_mixup', fallback=False)
    args.subtype = parser[key].get('subtype', fallback="")
    args.nb_classes = parser[key].getint('nb_classes', fallback=2)
    args.use_proj = parser[key].getboolean('use_proj', fallback=False)
    args.selection_type = parser['DATASET'].get('selection_type')
    args.mode = parser['DATASET'].get('mode')
    args.num_classes = parser['DATASET'].getint('num_classes')
    args.split = parser['DATASET'].get('split', fallback='idh')
    args.perceptual_weight = parser[key].getint('perceptual_weight', fallback=0)
    args.contr_weight = parser[key].getfloat('contr_weight', fallback=0.0)
    args.only_test_split = parser[key].getboolean('only_test_split', fallback=False)
    args.common_path = parser[key].get('common_path', fallback=None)
    args.checkpoint_perc = parser[key].get('checkpoint_perc', fallback=None)
    args.checkpoint_contr = parser[key].get('checkpoint_contr', fallback=None)
    args.use_only_test_dataset = parser[key].getboolean('use_only_test_dataset', fallback=False)
    args.use_imagenet = parser[key].getboolean('use_imagenet', fallback=False)
    args.use_edge_map = parser[key].getboolean('use_edge_map', fallback=True)
    args.volume_size = parser['DATASET'].getint('volume_size', fallback=96)
    args.use_barlow = parser['SETUP'].getboolean('use_barlow', fallback=False)
    args.fix_backbone = parser['FINE_TUNE_K_FOLD'].getboolean('fix_backbone', fallback=None)

    return args
