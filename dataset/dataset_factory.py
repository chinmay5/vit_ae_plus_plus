from dataset.brain_tumor import pretrain_tumor_data
from dataset.breast_cancer.breast_cancer_dataset_factory import build_dataset_from_breast_cancer_factory
from dataset.large_brats import large_brats_data
from dataset.lesions_dataset import lesion_data
from dataset.ms_lesion_dataset import ms_lesion_dataset


def get_dataset(dataset_name, mode, args, transforms=None, use_z_score=False, split='idh'):
    assert dataset_name in ['large_brats', 'brats', 'breast_cancer', 'lesion', 'ms_lesion'], f"Unsupported dataset {dataset_name}"
    if dataset_name == 'large_brats':
        return large_brats_data.create_the_dataset(mode=mode, split=args.split, args=args, transforms=transforms, use_z_score=use_z_score)
    elif dataset_name == 'brats':
        return pretrain_tumor_data.build_dataset(mode=mode, args=args, transforms=transforms, use_z_score=use_z_score)
    elif dataset_name == 'lesion':
        return lesion_data.build_dataset(mode, args=args, transforms=transforms, use_z_score=use_z_score)
    elif dataset_name == 'ms_lesion':
        return ms_lesion_dataset.build_dataset(mode, args=args, transforms=transforms, use_z_score=use_z_score)
    else:
        return build_dataset_from_breast_cancer_factory(selection_type=args.selection_type, mode=mode, transforms=transforms, use_z_score=use_z_score)