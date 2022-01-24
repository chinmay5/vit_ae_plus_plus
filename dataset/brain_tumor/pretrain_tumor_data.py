import os

import numpy as np
import torch
from torch import sqrt
from torch.utils.data import Dataset
import torchio as tio

BASE_PATH = '/mnt/cat/chinmay/brats_processed/data/splits'

class FlairData(Dataset):
    def __init__(self, filename='x_train_ssl.npy', transform=None, label_name=None, use_z_score=False):
        super(FlairData).__init__()
        file_path = os.path.join(BASE_PATH, filename)
        data_raw = np.load(file_path)
        self.data = data_raw.transpose([0, 4, 1, 2, 3])
        self.transform = transform
        self.use_z_score = use_z_score
        self.labels = np.load(os.path.join(BASE_PATH,label_name)) if label_name is not None else None
        print(f"Using z-score normalization: {use_z_score}")

    def __len__(self):
        return self.data.shape[0]

    def _normalize_data(self, volume):
        if self.use_z_score:
            # Since this is a single channel image so, we can ignore the `axis` parameter
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val)/ (max_val - min_val)
        return 2 * volume - 1  # Range of values [1, -1]

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val)/ (max_val - min_val)
        return volume  # Range of values [0, 1]

    def __getitem__(self, item):
        volume = torch.tensor(self.data[item], dtype=torch.float)
        # volume = self._min_max_normalize_data(volume)
        if self.transform is not None:
            volume = self.transform(volume)
        # If we normalize first and then apply transforms, the range of input values is changed to exceed the limits
        volume = self._normalize_data(volume)
        if self.labels is not None:
            return volume, torch.tensor(self.labels[item])
        return volume

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"


def build_dataset(mode, args=None, transforms=None, use_z_score=False):
    assert mode in ['train', 'valid', 'test', 'feat_extract'], "Invalid Mode selected"
    if mode == 'feat_extract':
        # A special case where we simply use all the data in our feature extraction pipeline. So, no augmentations
        # and the split file would be combination of both train and val
        return FlairData(filename='feature_extraction_x.npy', transform=None, label_name='feature_extraction_labels.npy', use_z_score=use_z_score)
    if mode == 'train':
        filename = 'x_train_ssl.npy'
        label_name = 'y_train_ssl.npy'
    elif mode == 'valid':
        filename = 'x_val_ssl.npy'
        label_name = 'y_val_ssl.npy'
    else:
        # Because of assert security net, this is test mode
        filename = 'feature_extraction_test_x.npy'
        label_name = 'feature_extraction_test_labels.npy'
    return FlairData(filename=filename, transform=transforms, label_name=label_name, use_z_score=use_z_score)



if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = FlairData(transform=transformations, use_z_score=True)
    sample = data[0]
    print(sample.shape)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0

    for batch_data in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
    # Also a check for other data splits
    train_data = build_dataset(mode='train')
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    all_ones, total = 0, 0
    for batch_data, labels in data_loader:
        all_ones += labels.sum()
        total += labels.shape[0]
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"% of ones {all_ones/total}")
    print(f"Max value is {max_val}, min value {min_val}")
    #     break
    # Checking for the validation split
    test_data = build_dataset(mode='feat_extract')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    for batch_data, labels in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")

