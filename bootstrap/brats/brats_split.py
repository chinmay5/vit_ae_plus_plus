import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split


base_dir = '/mnt/cat/chinmay/brats_processed'
file_path = os.path.join(base_dir, 'data', 'image', 'flair_all.npy')
data_raw = np.load(file_path)

save_folder = os.path.join(base_dir, 'data', 'splits')

file_locs = {"train_img_path": os.path.join(save_folder, 'x_train_ssl.npy'),
             "val_img_path": os.path.join(save_folder, 'x_val_ssl.npy'),
             "test_img_path": os.path.join(save_folder, 'x_test_ssl.npy'),
             "train_label_path": os.path.join(save_folder, 'y_train_ssl.npy'),
             "val_label_path": os.path.join(save_folder, 'y_val_ssl.npy'),
             "test_label_path": os.path.join(save_folder, 'y_test_ssl.npy')
             }


def sanity_check(train_split, val_split, test_split):
    train_set, val_set, test_set = set(train_split), set(val_split), set(test_split)
    length_arr = [len(train_set.intersection(val_set)), len(train_set.intersection(test_set)),
                  len(test_set.intersection(val_set))]
    return all([x == 0 for x in length_arr])


def split_brats_data():
    radiomics_path = os.path.join(base_dir, 'features_flair.npy')
    labels_path = os.path.join(base_dir, 'label_all.npy')

    radiomics = np.asarray(np.load(radiomics_path))
    labels = np.load(labels_path)
    indices = np.arange(radiomics.shape[0])
    # Getting indices of the train and test splits
    train_idx, test_idx = train_test_split(indices, train_size=0.9, stratify=labels, random_state=42)
    # Let us save some values for the validation split
    train_labels = labels[train_idx]
    train_idx, val_idx = train_test_split(train_idx, train_size=0.9, stratify=train_labels, random_state=42)
    sanity_check(train_idx, val_idx, test_idx)
    train_indices_save_path = os.path.join(base_dir, 'data', 'train_indices.npy')
    test_indices_save_path = os.path.join(base_dir, 'data', 'test_indices.npy')
    val_indices_save_path = os.path.join(base_dir, 'data', 'val_indices.npy')
    np.save(train_indices_save_path, train_idx)
    np.save(test_indices_save_path, test_idx)
    np.save(val_indices_save_path, val_idx)
    create_splits()


def create_splits():
    if any([os.path.exists(x) for x in file_locs.values()]):
        choice = input("Files exist. Please enter y to create new splits")
        if choice != 'y':
            print("Exiting")
            sys.exit(-1)
    train_indices = np.load(os.path.join(base_dir, 'data', 'train_indices.npy'))
    val_indices = np.load(os.path.join(base_dir, 'data', 'val_indices.npy'))
    test_indices = np.load(os.path.join(base_dir, 'data', 'test_indices.npy'))

    labels_path = os.path.join(base_dir, 'label_all.npy')
    all_labels = np.load(labels_path)
    # Get individual labels
    train_labels = all_labels[train_indices]
    val_labels = all_labels[val_indices]
    test_labels = all_labels[test_indices]
    # Get the features
    X_train_ssl = data_raw[train_indices]
    X_val_ssl = data_raw[val_indices]
    X_test_ssl = data_raw[test_indices]

    # We can use the train indices and create a validation split from them
    # Finally save the new splits
    print("Saving the new data splits")
    np.save(os.path.join(save_folder, 'x_train_ssl.npy'), X_train_ssl)
    np.save(os.path.join(save_folder, 'x_val_ssl.npy'), X_val_ssl)
    np.save(os.path.join(save_folder, 'x_test_ssl.npy'), X_test_ssl)
    np.save(os.path.join(save_folder, 'x_whole_ssl.npy'), data_raw)
    # Same for the labels
    np.save(os.path.join(save_folder, 'y_train_ssl.npy'), train_labels)
    np.save(os.path.join(save_folder, 'y_val_ssl.npy'), val_labels)
    np.save(os.path.join(save_folder, 'y_test_ssl.npy'), test_labels)
    np.save(os.path.join(save_folder, 'y_whole_ssl.npy'), all_labels)


if __name__ == '__main__':
    split_brats_data()
