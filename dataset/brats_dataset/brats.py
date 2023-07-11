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
        self.labels = np.load(os.path.join(BASE_PATH, label_name)) if label_name is not None else None
        print(f"Using z-score normalization: {use_z_score}")

    def __len__(self):
        return self.data.shape[0]

    def _normalize_data(self, volume):
        if self.use_z_score:
            # Since this is a single channel image so, we can ignore the `axis` parameter
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return 2 * volume - 1  # Range of values [1, -1]

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume  # Range of values [0, 1]

    def __getitem__(self, item):
        volume = torch.tensor(self.data[item], dtype=torch.float)
        original_volume = self._normalize_data(volume.clone())
        if self.transform is not None:
            volume = self.transform(volume)
        volume = self._normalize_data(volume)
        if self.labels is not None:
            return volume, original_volume, torch.tensor(self.labels[item])
        return volume, original_volume

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"


def build_dataset(mode, args=None, transforms=None, use_z_score=False):
    assert mode in ['train', 'val', 'test', 'whole'], f"Invalid Mode selected, {mode}"
    filename = f'x_{mode}_ssl.npy'
    label_name = f'y_{mode}_ssl.npy'
    return FlairData(filename=filename, transform=transforms, label_name=label_name, use_z_score=use_z_score)


if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = build_dataset(mode='whole')
    print(len(data))
    print(data[0][0].shape)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0

    for batch_data,_, label in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"shape -> {batch_data.shape}")
    print(f"Max value is {max_val}, min value {min_val}")
    # Also a check for other data splits
    train_data = build_dataset(mode='train')
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    all_ones, total = 0, 0
    for batch_data,_, labels in data_loader:
        all_ones += labels.sum()
        total += labels.shape[0]
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"% of ones {all_ones / total}")
    print(f"Max value is {max_val}, min value {min_val}")
    #     break
    # Checking for the validation split
    test_data = build_dataset(mode='train')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    for batch_data, _, labels in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
