import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

BASE_PATH = '/mnt/cat/chinmay/brats_processed/data'

file_path = os.path.join(BASE_PATH, 'image', 'flair_all.npy')
data_raw = np.load(file_path)
# Let us now save all the values.
save_folder = os.path.join(BASE_PATH, 'splits')
os.makedirs(save_folder, exist_ok=True)

file_locs = {"train_save_path":os.path.join(save_folder, 'x_train_ssl.npy'),
             "val_save_path": os.path.join(save_folder, 'x_val_ssl.npy'),
             "train_label_path": os.path.join(save_folder, 'y_train_ssl.npy'),
             "val_label_path": os.path.join(save_folder, 'y_val_ssl.npy')
             }

def create_splits():
    if any([os.path.exists(x) for x in file_locs.values()]):
        choice = input("Files exist. Please enter y to create new splits")
        if choice != 'y':
            print("Exiting")
            sys.exit(-1)
    train_indices = np.load(os.path.join(BASE_PATH, 'train_indices.npy'))
    # We can use the train indices and create a validation split from them
    all_labels = np.load(os.path.join('/mnt/cat/chinmay/brats_processed', 'label_all.npy'))
    train_labels = all_labels[train_indices]
    train_idx_ssl, val_idx_ssl = train_test_split(train_indices, train_size=0.85, stratify=train_labels, random_state=42)
    X_train_ssl = data_raw[train_idx_ssl]
    X_val_ssl = data_raw[val_idx_ssl]
    y_train_ssl = all_labels[train_idx_ssl]
    y_val_ssl = all_labels[val_idx_ssl]
    # Finally save the new splits
    print("Saving the new data splits")
    np.save(os.path.join(save_folder, 'x_train_ssl.npy'), X_train_ssl)
    np.save(os.path.join(save_folder, 'x_val_ssl.npy'), X_val_ssl)
    np.save(os.path.join(save_folder, 'y_train_ssl.npy'), y_train_ssl)
    np.save(os.path.join(save_folder, 'y_val_ssl.npy'), y_val_ssl)


if __name__ == '__main__':
    create_splits()
