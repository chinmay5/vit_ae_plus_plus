from functools import partial

from torch import nn

from model import vit_autoenc
from model.vit import VisionTransformer3D, VisionTransformer3DContrastive


def get_models(model_name, args):
    if model_name == 'autoenc':
        return vit_autoenc.__dict__[args.model](volume_size=args.volume_size, in_chans=args.in_channels, patch_size=args.patch_size)
    elif model_name == 'vit':
        return VisionTransformer3D(volume_size=args.volume_size, in_chans=args.in_channels, num_classes=args.nb_classes,
                                   patch_size=args.patch_size, global_pool=args.global_pool, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   drop_path_rate=args.drop_path)
    elif model_name == 'contrastive':
        return VisionTransformer3DContrastive(volume_size=args.volume_size, in_chans=args.in_channels, num_classes=args.nb_classes,
                                   patch_size=args.patch_size, global_pool=args.global_pool,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                   drop_path_rate=args.drop_path)
    else:
        raise NotImplementedError("Only AE model supported till now")
