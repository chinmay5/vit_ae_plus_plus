from dataset.brain_tumor import pretrain_tumor_data
from dataset.breast_cancer import pretrain_breast_cancer_data


def get_dataset(dataset_name, mode, args, transforms):
    assert dataset_name in ['brats', 'breast_cancer'], f"Unsupported dataset {dataset_name}"
    if dataset_name == 'brats':
        return pretrain_tumor_data.build_dataset(mode=mode, args=args, transforms=transforms)
    else:
        return pretrain_breast_cancer_data.build_dataset(mode=mode, transforms=transforms)
