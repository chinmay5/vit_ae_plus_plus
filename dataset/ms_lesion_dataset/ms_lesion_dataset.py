import os

import numpy as np
import pandas as pd
import torch
from torch import sqrt
from torch.utils.data import Dataset
import torchio as tio

from environment_setup import PROJECT_ROOT_DIR

BASE_PATH = '/mnt/cat/chinmay/ms_lesions_graph'
DATASET_SPLIT_FILES_PATH = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'ms_lesion_dataset')


class MSLesionDataset(Dataset):
    def __init__(self, mode, transform=None, use_z_score=False):
        super(MSLesionDataset).__init__()
        assert mode in ['ssl', 'test', 'whole'], f"Invalid model choice. Chosen: {mode}"
        data_raw = np.load(os.path.join(BASE_PATH, 'lesion_patches_gnn.npy'))
        self.data = data_raw.transpose([0, 4, 1, 2, 3])
        self.transform = transform
        self.use_z_score = use_z_score
        self.labels = pd.read_csv(os.path.join(DATASET_SPLIT_FILES_PATH, f'{mode}.csv'))
        print(f"Using z-score normalization: {use_z_score}")

    def __len__(self):
        return len(self.labels)

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
        idx, label = self.labels.iloc[item]['GlobalLesionID'].item(), self.labels.iloc[item]['New_Lesions_1y'].item()
        volume = torch.tensor(self.data[idx], dtype=torch.float)
        original_volume = self._normalize_data(volume.clone())
        if self.transform is not None:
            volume = self.transform(volume)
        volume = self._normalize_data(volume)
        return volume, original_volume, torch.tensor(label)

    def __str__(self):
        return f"Two channel MRI data with transforms = {self.transform}"


def build_dataset(mode, args=None, transforms=None, use_z_score=False):
    return MSLesionDataset(mode=mode, transform=transforms, use_z_score=use_z_score)


if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = build_dataset(mode='train')
    print(len(data))
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0

    for batch_data, _, label in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
    # Also a check for other data splits
    train_data = build_dataset(mode='val')
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    all_ones, total = 0, 0
    for batch_data, _, labels in data_loader:
        all_ones += labels.sum()
        total += labels.shape[0]
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"% of ones {all_ones / total}")
    print(f"Max value is {max_val}, min value {min_val}")
