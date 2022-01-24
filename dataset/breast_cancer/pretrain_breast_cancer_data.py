import os
import pickle

import numpy as np
import torch
from torch import sqrt
from torch.utils.data import Dataset
import torchio as tio

BASE_PATH = os.path.join('/', 'mnt', 'cat', 'chinmay', 'duke_breast_cancer', 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203')
img_path = os.path.join(BASE_PATH, 'cropped_images')
DATA_SPLIT_INDICES_PATH = os.path.join(BASE_PATH, 'data_splits')
# Labels are the same for both cases hence we simply re-use the `label` pickles rather than generating them from scratch
RADIOMICS_SAVE_FILE_PATH = os.path.join(BASE_PATH, 'radiomics_feat')


class FlairData(Dataset):
    def __init__(self, mode, transform=None, use_z_score=False):
        super(FlairData).__init__()
        self.filenames = self.load_index(mode)
        self.labels_dict = pickle.load(open(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'labels_dict.npy'), 'rb'))
        self.transform = transform
        self.use_z_score= use_z_score

    def __len__(self):
        return len(self.filenames)

    def _normalize_data(self, volume):
        if self.use_z_score:
            # Since this is a single channel image so, we can ignore the `axis` parameter
            return (volume - volume.mean()) / sqrt(volume.var())
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val)/ (max_val - min_val)
        return 2 * volume - 1  # Range of values [1, -1]

    def __getitem__(self, item):
        file_name = self.filenames[item]
        volume = torch.as_tensor(np.load(os.path.join(img_path, f"{file_name}.npy")), dtype=torch.float)
        # We add channel dimension to the input. Perhaps a step missed during pre-processing
        volume.unsqueeze_(0)
        if self.transform is not None:
            volume = self.transform(volume)
        # If we normalize first and then apply transforms, the range of input values is changed to exceed the limits
        volume = self._normalize_data(volume)
        return volume, torch.tensor(self.labels_dict[file_name])

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"

    def load_index(self, mode):
        # We take care of the case wherein we need to combine both train and val sets for feature extraction
        if mode == 'feat_extract':
            train_split_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'train.pkl'), 'rb'))
            val_split_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'val.pkl'), 'rb'))
            train_split_indices.extend(val_split_indices)
            return train_split_indices
        elif mode == 'train':
            train_split_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'train.pkl'), 'rb'))
            return train_split_indices
        elif mode == 'val':
            val_split_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'val.pkl'), 'rb'))
            return val_split_indices
        else:
            test_split_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'test.pkl'), 'rb'))
            return test_split_indices


def build_dataset(mode, args=None, transforms=None, use_z_score=False):
    assert mode in ['train', 'valid', 'test', 'feat_extract'], "Invalid Mode selected"
    return FlairData(mode=mode, transform=transforms, use_z_score=use_z_score)



if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = FlairData(mode='feat_extract', transform=transformations)
    sample = data[0][0]
    print(sample.shape)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0

    for batch_data, _ in data_loader:
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
    test_data = build_dataset(mode='test')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    for batch_data, labels in data_loader:
        if batch_data.max() > max_val:
            max_val = batch_data.max()
        if batch_data.min() < min_val:
            min_val = batch_data.min()
    print(f"Max value is {max_val}, min value {min_val}")

