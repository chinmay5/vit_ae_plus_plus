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
    args.cross_entropy_wt = torch.as_tensor(read_float_lists(parser=parser, key='DATASET', variable='cross_entropy_wt'), dtype=torch.float)
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
    return args
