import glob
import os
import pickle
import random

import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR

BASE_DIR = os.path.join('/mnt', 'cat', 'chinmay', 'duke_breast_cancer')
META_DATA_DIR = os.path.join(BASE_DIR, 'data_files')
MRI_IMG_DIR = os.path.join(BASE_DIR, 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203', 'Duke-Breast-Cancer-MRI')
SAVE_PATH = os.path.join(BASE_DIR, 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203', 'cropped_images')
SPLIT_SAVE_FILE_PATH = os.path.join(BASE_DIR, 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203', 'data_splits')
RADIOMICS_SAVE_FILE_PATH = os.path.join(BASE_DIR, 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203', 'radiomics_feat')
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SPLIT_SAVE_FILE_PATH, exist_ok=True)
os.makedirs(RADIOMICS_SAVE_FILE_PATH, exist_ok=True)

required_cols = ['Patient ID', 'Lymphadenopathy or Suspicious Nodes']
converters = {
    'Patient ID': str,
    'Lymphadenopathy or Suspicious Nodes': int
}


def see_all_cols(filename):
    df = pd.read_excel(filename, engine='openpyxl', skiprows=[0])
    print(df.columns)


def read_file(filename):
    # For the Clinical_and_Other_Features.xslx, we need to skip two rows and stick to specified columns.
    # df = pd.read_excel(filename, engine='openpyxl', skiprows=[0, 2], usecols=required_cols, converters=converters)
    df = pd.read_excel(filename, engine='openpyxl',
                       index_col=0)  # An extra column appears otherwise. Known bug. Fixed in new version of pandas
    return df


DESIRED_SHAPE = [96, 96, 96]
df = read_file(filename=os.path.join(META_DATA_DIR, 'Annotation_Boxes.xlsx'))
bad_files = []
filename_filter_keys = ['Ph1', 'Ph2', 'Ph3', 'Ph4', 't1', 'T1']


def process_one_dimension(dimension_x, min_x, max_x, index=0):
    """
    :param dimension_x: Indicates the extent of the given tumor in single dimension
    :param min_x: max value of coordinate
    :param max_x: min value of coordinate
    :param index: axis we are talking about, [x,y,z]
    :return: min_val, max_val: The cropping dimensions
    """
    desired_shape_extent = DESIRED_SHAPE[index]
    mid_x = (max_x + min_x) // 2
    if dimension_x < desired_shape_extent:
        # The ROI is smaller than the slice
        min_val, max_val = int(mid_x - desired_shape_extent / 2), int(mid_x + desired_shape_extent / 2)
        min_val = 0 if min_val < 0 else min_val  # Clamping value
        return min_val, max_val
    else:
        print("Tumor extent is big. Starting from the top and taking the max value")
        return max_x - desired_shape_extent, max_x


def obtain_tumor_dims(start_row, end_row, start_col, end_col, start_slice, end_slice):
    dimension_x, dimension_y, dimension_z = end_row - start_row, end_col - start_col, end_slice - start_slice
    assert dimension_x > 0 and dimension_y > 0 and dimension_z > 0, "Invalid selection of slice dimensions"
    row_begin, row_end = process_one_dimension(dimension_x=dimension_x, min_x=start_row, max_x=end_row, index=0)
    col_begin, col_end = process_one_dimension(dimension_x=dimension_y, min_x=start_col, max_x=end_col, index=1)
    slice_begin, slice_end = process_one_dimension(dimension_x=dimension_z, min_x=start_slice, max_x=end_slice, index=2)
    return row_begin, row_end, col_begin, col_end, slice_begin, slice_end


def read_dicoms_as_np_array(dicom_files, dicom_location):
    combined_arr = []
    for dicom_file in dicom_files:
        combined_arr.append(pydicom.dcmread(os.path.join(dicom_location, dicom_file)).pixel_array)
    return np.stack(combined_arr)


def select_file(scan_name, include_pre=False):
    folder_path = os.path.join(MRI_IMG_DIR, scan_name, '*', '*')
    all_files = glob.glob(folder_path)
    if not include_pre:
        removal_vals = [x for x in all_files for skip_tag in filename_filter_keys if skip_tag in x]
        selected_folder = [x for x in all_files if x not in removal_vals]
    else:
        selected_folder = [x for x in all_files if 'pre' in x]
    if len(selected_folder) != 1:
        bad_files.append(scan_name)
        raise AttributeError("Multiple candidate files. Please check scan_type criterion.")
    selected_folder = selected_folder[0]
    dicom_location = os.path.join(folder_path, selected_folder)
    dicom_files = os.listdir(os.path.join(folder_path, selected_folder))
    sorted(dicom_files, key=lambda x: x[2: x.find(".dcm")])
    scan = read_dicoms_as_np_array(dicom_files=dicom_files, dicom_location=dicom_location)
    return scan


def sanity_check_fur_scan_shapes():
    """
    Method to make sure that all the scans are of the shape 96, 96, 96
    :return: None
    """
    for item in os.listdir(SAVE_PATH):
        assert np.all(np.load(os.path.join(SAVE_PATH, item)).shape == np.array([96, 96, 96])), f"Invalid shape for {item}"

def process_single_scan(scan_name, triage=True, include_pre=False):
    row_begin, row_end, col_begin, col_end, slice_begin, slice_end = df.loc[scan_name]
    row_begin, row_end, col_begin, col_end, slice_begin, slice_end = obtain_tumor_dims(start_row=row_begin,
                                                                                       end_row=row_end,
                                                                                       start_col=col_begin,
                                                                                       end_col=col_end,
                                                                                       start_slice=slice_begin,
                                                                                       end_slice=slice_end)

    mri_scan = select_file(scan_name=scan_name, include_pre=include_pre)
    x_dim, y_dim, z_dim = mri_scan.shape
    cropped_scan = mri_scan[slice_begin:slice_end, row_begin:row_end, col_begin:col_end]
    print(cropped_scan.shape)
    if slice_end > x_dim:
        pad_length_x = slice_end - x_dim
        cropped_scan = np.pad(cropped_scan, [(0, pad_length_x), (0, 0), (0, 0)])
    if cropped_scan.shape[0] != 96:
        pad_length_x = 96 - cropped_scan.shape[0]
        cropped_scan = np.pad(cropped_scan, [(0, pad_length_x), (0, 0), (0, 0)])
    # assert slice_end <= x_dim and row_end <= y_dim and col_end < z_dim, f"Something wrong with the orientation for {scan_name} with {slice_end, row_end, col_end} while its shape is {mri_scan.shape}"
    assert np.all(cropped_scan.shape == np.array([96, 96, 96])), f"Something wrong with the orientation for {scan_name} with begin {slice_begin, row_begin, col_begin} end {slice_end, row_end, col_end} while its shape is {mri_scan.shape} and crop shape {cropped_scan.shape}"
    if triage:
        print(cropped_scan.shape)
        return
    np.save(os.path.join(SAVE_PATH, scan_name), cropped_scan.astype(np.float))


def process_all_scans():
    for scan in tqdm(df.index):
        try:
            process_single_scan(scan_name=scan, triage=False)
        except AttributeError as e:
            print(f"Scan :- {scan} is bad!!!")
    pickle.dump(bad_files, open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'wb'))


def handle_bad_files_from_first_round():
    """
    Many files have only the prefix "pre" in their name and not phases. We handle such scans in this round.
    :return:
    """
    print("Handling the remaining examples from the first round")
    bad_files = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'rb'))
    for scan in tqdm(bad_files):
        try:
            process_single_scan(scan_name=scan, triage=False, include_pre=True)
        except AttributeError as e:
            print(f"Scan :- {scan} is bad!!!")


def sanity_check(train_split, val_split, test_split):
    train_set, val_set, test_set = set(train_split), set(val_split), set(test_split)
    length_arr = [len(train_set.intersection(val_set)), len(train_set.intersection(test_set)),
                  len(test_set.intersection(val_set))]
    return all([x == 0 for x in length_arr])


def read_radiomics_labels():
    filename = os.path.join(META_DATA_DIR, 'Clinical_and_Other_Features.xlsx')
    label_df = pd.read_excel(filename, engine='openpyxl', skiprows=[0, 2], usecols=required_cols, converters=converters,
                             index_col=0)
    labels_dict = {}
    for scan_name in tqdm(label_df.index):
        labels_dict[scan_name] = label_df.loc[scan_name].item()
    return labels_dict


def save_radiomics_data():
    filename = os.path.join(META_DATA_DIR, 'Imaging_Features.xlsx')
    df = pd.read_excel(filename, engine='openpyxl',
                       index_col=0)
    radiomics_features_dict = {}
    for scan_name in tqdm(df.index):
        radiomics_features_dict[scan_name] = df.loc[scan_name].values
    pickle.dump(radiomics_features_dict, open(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'radiomics_all.pkl'), 'wb'))

def pruge_nan_rows(feat_arr, labels_arr):
    # We should remove both feature and label entries for nan
    print(f"Original sizes are: features: {feat_arr.shape} and labels {labels_arr.shape}")
    nan_indices = np.isnan(feat_arr).any(axis=1)
    feat_arr, labels_arr = feat_arr[~nan_indices], labels_arr[~nan_indices]
    print(f"Sizes after purging features: {feat_arr.shape} and labels {labels_arr.shape}")
    return feat_arr, labels_arr

def train_val_radiomics_feat():
    labels_dict = read_radiomics_labels()
    radiomics_features_dict = pickle.load(open(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'radiomics_all.pkl'), 'rb'))
    train_split_indices = pickle.load(open(os.path.join(SPLIT_SAVE_FILE_PATH, 'train.pkl'), 'rb'))
    val_split_indices = pickle.load(open(os.path.join(SPLIT_SAVE_FILE_PATH, 'val.pkl'), 'rb'))
    test_split_indices = pickle.load(open(os.path.join(SPLIT_SAVE_FILE_PATH, 'test.pkl'), 'rb'))
    # The indices have items of the form 'Breast_MRI_208.npy', we just need the patientId. Hence,
    train_split_indices = [x[:x.find('.npy')] for x in train_split_indices]
    val_split_indices = [x[:x.find('.npy')] for x in val_split_indices]
    test_split_indices = [x[:x.find('.npy')] for x in test_split_indices]
    # Now we store values in an array
    # Let us remove all the indices that can lead to nan values in its shape
    train_numpy_feat, train_numpy_labels, train_surviving_indices = purge_nans(labels_dict=labels_dict,
                                                                               radiomics_features_dict=radiomics_features_dict,
                                                                               split_indices=train_split_indices)
    # Similarly for the validation elements
    val_numpy_feat, val_numpy_labels, val_surviving_indices = purge_nans(labels_dict=labels_dict,
                                                                               radiomics_features_dict=radiomics_features_dict,
                                                                               split_indices=val_split_indices)
    # Similarly for the test elements
    test_numpy_feat, test_numpy_labels, test_surviving_indices = purge_nans(labels_dict=labels_dict,
                                                                         radiomics_features_dict=radiomics_features_dict,
                                                                         split_indices=test_split_indices)
    # Old code block. Hopefully, task already taken care of
    # Purge the nan entries
    # train_numpy_feat, train_numpy_labels = pruge_nan_rows(feat_arr=train_numpy_feat, labels_arr=train_numpy_labels)
    # val_numpy_feat, val_numpy_labels = pruge_nan_rows(feat_arr=val_numpy_feat, labels_arr=val_numpy_labels)
    # test_numpy_feat, test_numpy_labels = pruge_nan_rows(feat_arr=test_numpy_feat, labels_arr=test_numpy_labels)
    # Now save all these scans
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_feat.npy'), train_numpy_feat)
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_labels.npy'), train_numpy_labels)
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'val_feat.npy'), val_numpy_feat)
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'val_labels.npy'), val_numpy_labels)
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_feat.npy'), test_numpy_feat)
    np.save(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'), test_numpy_labels)
    # Also store the labels dictionary since it would make our lives easier
    pickle.dump(labels_dict, open(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'labels_dict.npy'),'wb'))
    # Finally, let us also update the label indices since we want to have the same consistant ones across radiomics
    # and SSL
    pickle.dump(train_surviving_indices, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'train.pkl'), 'wb'))
    pickle.dump(val_surviving_indices, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'val.pkl'), 'wb'))
    pickle.dump(test_surviving_indices, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'test.pkl'), 'wb'))


def purge_nans(labels_dict, radiomics_features_dict, split_indices):
    numpy_feat = []
    numpy_labels = []
    surviving_indices = []
    for idx in split_indices:
        features = radiomics_features_dict[idx]
        if np.any(np.isnan(features)):
            continue
        surviving_indices.append(idx)
        numpy_feat.append(radiomics_features_dict[idx])
        numpy_labels.append(labels_dict[idx])
    numpy_feat = np.stack(numpy_feat)
    numpy_labels = np.stack(numpy_labels)
    return numpy_feat, numpy_labels, surviving_indices


def train_val_test_splits():
    files = os.listdir(os.path.join(SAVE_PATH))
    random.shuffle(files)
    total_len = len(files)
    train_len = int(0.8 * total_len)
    train_split, test_split = files[:train_len], files[train_len:]
    # Now, we create val split from the train_split
    val_len = int(train_len * 0.1)
    random.shuffle(train_split)
    train_split, val_split = train_split[val_len:], train_split[:val_len]
    assert sanity_check(train_split=train_split, val_split=val_split,
                        test_split=test_split), "Splitting process caused overlap. Aborting!"
    # Now we can save the elements
    print(
        f"Creating dataset with train_len {len(train_split)}, val_len {len(val_split)} and test_len {len(test_split)}")
    pickle.dump(train_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'train.pkl'), 'wb'))
    pickle.dump(val_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'val.pkl'), 'wb'))
    pickle.dump(test_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'test.pkl'), 'wb'))


def bootstrap_setup():
    train_val_test_splits()
    save_radiomics_data()
    train_val_radiomics_feat()


if __name__ == '__main__':
    # tumor_data_file = os.path.join(META_DATA_DIR, 'Clinical_and_Other_Features.xlsx')
    # read_file(tumor_data_file)
    process_all_scans()
    handle_bad_files_from_first_round()
    sanity_check_fur_scan_shapes()
    # process_single_scan(scan_name='Breast_MRI_001', triage=False)
    # pickle.dump(bad_files, open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'wb'))

    # pickle.dump(bad_files, open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'wb'))
    bootstrap_setup()
