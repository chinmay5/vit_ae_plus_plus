import os

import numpy as np
import SimpleITK as sitk
import pandas as pd
from sklearn.model_selection import train_test_split

from environment_setup import PROJECT_ROOT_DIR

ROOT_DIR = os.path.join("/", "mnt", "cat", "chinmay", "lesions")
SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "lesions")

os.makedirs(SPLIT_SAVE_FILE_PATH, exist_ok=True)
LABEL_MAP = {
    "Enhancing": 1,
    "Young": 0
}

def read_one_scan():
    whole_scan = np.load(os.path.join(ROOT_DIR, 'lesion_array_bran.npy'))
    print(whole_scan.shape)
    # The resulting shape was (32, 32, 32, 2, 3339)
    one_scan = whole_scan[..., 0]
    one_scan = one_scan.transpose([3, 0, 1, 2])
    print(one_scan.shape)
    result_image = sitk.GetImageFromArray(one_scan[0])
    # write the image
    sitk.WriteImage(result_image, 'result.nii.gz')


def check_sanity(train_idx, test_idx, labels_df):
    train_idx_set = set(train_idx.index)
    test_idx_set = set(test_idx.index)
    # Assert that there are no label leakage
    assert len(train_idx_set.intersection(test_idx_set)) == 0
    train_labels = labels_df.iloc[train_idx.index]
    test_labels = labels_df.iloc[test_idx.index]
    print(train_labels[['labels']].value_counts())
    print(test_labels[['labels']].value_counts())


def read_labels():
    np.random.seed(42)
    labels_df = pd.read_csv(os.path.join(ROOT_DIR, 'lesion_patches_bran.csv'), skiprows=0)
    labels_df["labels"] = labels_df["Class"].map(LABEL_MAP)
    labels_df = labels_df.drop('Class', axis=1)
    train_idx, test_idx = train_test_split(labels_df, train_size=0.9, stratify=labels_df[['labels']], random_state=42)
    check_sanity(train_idx, test_idx, labels_df)
    # Let us also store the whole label mode, which might be needed for some feature extraction
    labels_df.to_csv(os.path.join(SPLIT_SAVE_FILE_PATH, 'whole.csv'), index=False)
    train_idx.to_csv(os.path.join(SPLIT_SAVE_FILE_PATH, 'train.csv'), index=False)
    test_idx.to_csv(os.path.join(SPLIT_SAVE_FILE_PATH, 'test.csv'), index=False)


if __name__ == '__main__':
    # read_one_scan()
    read_labels()
