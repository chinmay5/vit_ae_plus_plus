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

class BaseData(Dataset):
    def __init__(self, mode, transform=None, use_z_score=False):
        super(BaseData).__init__()
        self.filenames = self.load_index(mode)
        self.transform = transform
        self.use_z_score= use_z_score
        self.only_scans = (mode == 'ssl')

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
        label = -100
        if self.only_scans:
            file_name = self.filenames[item]
        else:
            file_name, label = self.filenames[item]
        volume = torch.as_tensor(np.load(os.path.join(img_path, f"{file_name}.npy")), dtype=torch.float)
        # We add channel dimension to the input. Perhaps a step missed during pre-processing
        volume.unsqueeze_(0)
        original_volume = volume.clone()
        if self.transform is not None:
            volume = self.transform(volume)
        # If we normalize first and then apply transforms, the range of input values is changed to exceed the limits
        volume = self._normalize_data(volume)
        return volume, original_volume, torch.tensor(label)

    def __str__(self):
        return f"Pre-train Flair MRI data with transforms = {self.transform}"

    def load_index(self, mode):
        raise NotImplementedError


class FlairDataClinical(BaseData):
    def __init__(self, mode, transform=None, use_z_score=False):
        super(FlairDataClinical, self).__init__(mode=mode, transform=transform, use_z_score=use_z_score)

    def load_index(self, mode):
        # We take care of the case wherein we need to combine both train and val sets for feature extraction
        if mode == 'ssl':
            ssl_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'clinical_ssl.pkl'), 'rb'))
            return ssl_indices
        elif mode == 'labels':
            return pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'clinical_annotated_mit_labels.pkl'), 'rb'))
        else:
            raise AttributeError(f"Invalid choice of mode. Chosen {mode}")


class FlairDataPathology(BaseData):
    def __init__(self, mode, transform=None, use_z_score=False):
        super(FlairDataPathology, self).__init__(mode=mode, transform=transform, use_z_score=use_z_score)

    def load_index(self, mode):
        # We take care of the case wherein we need to combine both train and val sets for feature extraction
        if mode == 'ssl':
            ssl_indices = pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'pathology_ssl.pkl'), 'rb'))
            return ssl_indices
        elif mode == 'labels':
            return pickle.load(open(os.path.join(DATA_SPLIT_INDICES_PATH, 'pathology_annotated_mit_labels.pkl'), 'rb'))
        else:
            raise AttributeError(f"Invalid choice of mode. Chosen {mode}")



def build_dataset(selection_type, mode, args=None, transforms=None, use_z_score=False):
    assert mode in ['ssl', 'labels'], "Invalid Mode selected"
    if selection_type == 'clinical':
        return FlairDataClinical(mode=mode, transform=transforms, use_z_score=use_z_score)
    elif selection_type == 'pathology':
        return FlairDataPathology(mode=mode, transform=transforms, use_z_score=use_z_score)




if __name__ == '__main__':
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)

    data = FlairDataPathology(mode='ssl', transform=transformations)
    sample = data[0][0]
    print(sample.shape)
    data_loader = torch.utils.data.DataLoader(data, batch_size=4)
    min_val, max_val = float("inf"), 0
    try:
        for batch_data, _, _ in data_loader:
            if batch_data.max() > max_val:
                max_val = batch_data.max()
            if batch_data.min() < min_val:
                min_val = batch_data.min()
    except FileNotFoundError as e:
        print(e)
    print(f"Max value is {max_val}, min value {min_val}")
    # Also a check for other data splits
    train_data = build_dataset(selection_type='pathology', mode='labels')
    data_loader = torch.utils.data.DataLoader(train_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    label_set = set()
    try:
        for batch_data, _, labels in data_loader:
            label_set.update(labels.numpy().tolist())
            if batch_data.max() > max_val:
                max_val = batch_data.max()
            if batch_data.min() < min_val:
                min_val = batch_data.min()
    except FileNotFoundError as e:
        print(e)
    print(label_set)
    #     break
    # Checking for the validation split
    test_data = build_dataset(selection_type='clinical', mode='ssl')
    data_loader = torch.utils.data.DataLoader(test_data, batch_size=16)
    min_val, max_val = float("inf"), 0
    try:
        for batch_data, _, labels in data_loader:
            if batch_data.max() > max_val:
                max_val = batch_data.max()
            if batch_data.min() < min_val:
                min_val = batch_data.min()
    except FileNotFoundError as e:
        print(e)
    print(f"Max value is {max_val}, min value {min_val}")

