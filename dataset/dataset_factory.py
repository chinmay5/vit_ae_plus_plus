from dataset.brats_dataset import brats
from dataset.egd_dataset import egd


def get_dataset(dataset_name, mode, args, transforms=None, use_z_score=False, split='idh'):
    assert dataset_name in ['egd', 'brats'], f"Unsupported dataset {dataset_name}"
    if dataset_name == 'egd':
        return egd.build_dataset(mode=mode, split=args.split, args=args, transforms=transforms, use_z_score=use_z_score)
    else:
        return brats.build_dataset(mode=mode, args=args, transforms=transforms, use_z_score=use_z_score)
