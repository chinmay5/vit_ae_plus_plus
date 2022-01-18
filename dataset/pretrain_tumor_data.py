import os

import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio

BASE_PATH = '/mnt/cat/chinmay/brats_processed/data/splits'

class FlairData(Dataset):
    def __init__(self, filename='x_train_ssl.npy', transform=None, label_name=None):
        super(FlairData).__init__()
        file_path = os.path.join(BASE_PATH, filename)
        data_raw = np.load(file_path)
        self.data = data_raw.transpose([0, 4, 1, 2, 3])
        self.transform = transform
        self.labels = np.load(os.path.join(BASE_PATH,label_name)) if label_name is not None else None

    def __len__(self):
        return self.data.shape[0]

    def _normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val)/ (max_val - min_val)
        return 2 * volume - 1  # Range of values [1, -1]

    def __getitem__(self, item):
        volume = torch.tensor(self.data[item], dtype=torch.float)
        if self.transform is not None:
            volume = self.transform(volume)
        # If we normalize first and then apply transforms, the range of input values is changed to exceed the limits
        volume = self._normalize_data(volume)
        if self.labels is not None:
            return volume, torch.tensor(self.labels[item])
        return volume

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"


def build_dataset(is_train, args=None, transforms=None):
    filename = 'x_train_ssl.npy' if is_train else 'x_val_ssl.npy'
    label_name = 'y_train_ssl.npy' if is_train else 'y_val_ssl.npy'
    return FlairData(filename=filename, transform=transforms, label_name=label_name)



if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = FlairData(transform=transformations)
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
    # train_data = build_dataset(is_train=True)
    # data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    # for batch_data, labels in data_loader:
    #     print(batch_data.shape)
    #     print(labels.dtype)
    #     break
    # Checking for the validation split
    # val_data = build_dataset(is_train=False)
    # data_loader = torch.utils.data.DataLoader(val_data, batch_size=16)
    # for batch_data, labels in data_loader:
    #     print(batch_data.shape)
    #     print(labels.dtype)
    #     break

