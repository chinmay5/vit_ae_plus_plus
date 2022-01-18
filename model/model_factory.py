from model import vit_autoenc
from model.vit import VisionTransformer3D


def get_models(model_name, args):
    if model_name == 'autoenc':
        return vit_autoenc.__dict__[args.model](volume_size=args.volume_size, in_chans=args.in_channels, patch_size=args.patch_size)
    elif model_name == 'vit':
        return VisionTransformer3D(volume_size=args.volume_size, in_chans=args.in_channels, num_classes=args.nb_classes, patch_size=args.patch_size)
    else:
        raise NotImplementedError("Only AE model supported till now")
