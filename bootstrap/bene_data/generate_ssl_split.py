import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR

ROOT_DIR = os.path.join("/", "mnt", "cat", "chinmay", "glioma_Bene")
SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "large_brats")

target_col = 'who_idh_mutation_status'
label_converter = {
    'Subject': str,
    target_col: int
}
usecols = ['Subject', target_col]


def choose_valid(img_path, mri_scans, has_labels):
    valid_scans = []
    for scan in mri_scans:
        try:
            if has_labels:
                scan, label = scan
            _ = np.load(os.path.join(img_path, scan, "flair.npy"))
            _ = np.load(os.path.join(img_path, scan, "t1ce.npy"))
            _ = np.load(os.path.join(img_path, scan, "t1.npy"))
            _ = np.load(os.path.join(img_path, scan, "t2.npy"))
            valid_scans.append((scan, label)) if has_labels else valid_scans.append(scan)
        except FileNotFoundError as e:
            print(f"Skipping!!! {e}")
    return valid_scans


def read_custom_labels():
    filename = os.path.join(ROOT_DIR, 'bwiestler_1_26_2022_16_29_9.csv')
    label_df = pd.read_csv(filename, index_col=0, usecols=usecols)
    labels_dict = {}
    for scan_name in tqdm(label_df.index):
        labels_dict[scan_name] = label_df.loc[scan_name].item()
    return labels_dict


def get_ssl_items(filename):
    labels_dict = read_custom_labels()
    # We get nan values as the missing entries. We can filter away the missing values now.
    ssl_mri_scans = []
    downstream_scans = []
    for name, label in labels_dict.items():
        if np.isnan(label):
            raise AttributeError("Something is wrong")
        if label == -1:
            ssl_mri_scans.append(f"MR_{name}")
        else:
            downstream_scans.append((f"MR_{name}", label))
    # Some sanity checks
    ssl_set, downstream_set = set(ssl_mri_scans), set([x[0] for x in downstream_scans])
    assert len(ssl_set.intersection(downstream_set)) == 0, "Something wrong with the splitting, Aborting"
    print(f"Length of SSL split {len(ssl_set)}")
    print(f"Length of Supervised split {len(downstream_set)}")
    # Let us quickly remove suuch scans that we had issues in pre-processing
    # print("Processing SSL Files")
    ssl_mri_scans = choose_valid(os.path.join(ROOT_DIR, 'pre_processed'), ssl_mri_scans, has_labels=False)
    # print("Processing Label Files")
    downstream_scans = choose_valid(os.path.join(ROOT_DIR, 'pre_processed'), downstream_scans, has_labels=True)
    print(f"Length of SSL split {len(ssl_mri_scans)}")
    print(f"Length of Supervised split {len(downstream_scans)}")
    pickle.dump(ssl_mri_scans, open(os.path.join(SPLIT_SAVE_FILE_PATH, f'{filename}_ssl.pkl'), 'wb'))
    pickle.dump(downstream_scans,
                open(os.path.join(SPLIT_SAVE_FILE_PATH, f'{filename}_annotated_mit_labels.pkl'), 'wb'))


if __name__ == '__main__':
    get_ssl_items(filename='who_idh_mutation_status')
