import os
import pickle

import numpy as np
import torch
from torch import sqrt
from torch.utils.data import Dataset
import torchio as tio

from environment_setup import PROJECT_ROOT_DIR

BASE_PATH = '/mnt/cat/chinmay/glioma_Bene/pre_processed'


class EGDData(Dataset):
    def __init__(self, mode, transform=None, use_z_score=False, split='idh'):
        super(EGDData).__init__()
        split_path = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'egd_dataset')
        self.idh_mode_filename_map = {
            "ssl": 'who_idh_mutation_status_ssl.pkl',
            "test": 'who_idh_mutation_status_annotated_mit_labels.pkl',
            "whole": "who_idh_mutation_status_all.pkl",
            "whole_train": "who_idh_mutation_status_all_train.pkl"
        }
        filename = self.get_filename(mode, split=split)
        self.indices = pickle.load(open(os.path.join(split_path, filename), 'rb'))
        self.transform = transform
        self.use_z_score = use_z_score
        self.has_labels = mode == 'test'
        print(f"Using z-score normalization: {use_z_score}")

    def get_filename(self, mode, split='idh'):
        if split == 'idh':
            filename = self.idh_mode_filename_map[mode]
        elif split == '1p19q':
            filename = 'who_1p19q_codeletion_ssl.pkl' if mode == 'ssl' else 'correct_who_1p19q_codeletion_annotated_mit_labels.pkl'
        else:
            raise AttributeError("Invalid split selected")
        return filename

    def __len__(self):
        return len(self.indices)

    def _normalize_data(self, volume):
        if self.use_z_score:
            return (volume - torch.mean(volume, dim=[1, 2, 3], keepdim=True)) / sqrt(
                torch.var(volume, dim=[1, 2, 3], keepdim=True))
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return 2 * volume - 1  # Range of values [1, -1]

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume  # Range of values [0, 1]

    def load_volume(self, scan_name):
        flair = np.load(os.path.join(BASE_PATH, scan_name, "flair.npy"))
        t1ce = np.load(os.path.join(BASE_PATH, scan_name, "t1ce.npy"))
        t1 = np.load(os.path.join(BASE_PATH, scan_name, "t1.npy"))
        t2 = np.load(os.path.join(BASE_PATH, scan_name, "t2.npy"))
        volume = np.stack([flair, t1ce, t1, t2])
        return volume

    def __getitem__(self, item):
        if self.has_labels:
            volume = torch.tensor(self.load_volume(self.indices[item][0]), dtype=torch.float)
            label = torch.tensor(self.indices[item][1])
        else:
            volume = torch.tensor(self.load_volume(self.indices[item]), dtype=torch.float)
            label = torch.tensor(-100)  # Since collate_fn does not work with `None`
        original_volume = self._normalize_data(volume.clone())
        if self.transform is not None:
            volume = self.transform(volume)
        volume = self._normalize_data(volume)
        return volume, original_volume, label

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"


def build_dataset(mode, split, args=None, transforms=None, use_z_score=False):
    assert mode in ['ssl', 'test', 'whole', 'whole_train'], f"Invalid Mode selected, {mode}"
    return EGDData(mode=mode, split=split, transform=transforms, use_z_score=use_z_score)


if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = build_dataset(mode='whole', split='idh')
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
    train_data = build_dataset(mode='test', split='1p19q')
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
    #     break
    # Checking for the validation split
    test_data = build_dataset(mode='ssl', use_z_score=True, split='idh')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    for batch_data, _, labels in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")
