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
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SPLIT_SAVE_FILE_PATH, exist_ok=True)

required_cols = ['Patient ID', 'Mol Subtype']
converters = {
    'Patient ID': str,
    'Mol Subtype': int
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
    desired_shape_extent = DESIRED_SHAPE[index]
    mid_x = (max_x + min_x) // 2
    if dimension_x < desired_shape_extent:
        # The ROI is smaller than the slice
        min_val, max_val = int(mid_x - desired_shape_extent / 2), int(mid_x + desired_shape_extent / 2)
        min_val = 0 if min_val < 0 else min_val  # Clamping value
        return min_val, max_val
    else:
        print("Tumor extent is big. Starting from the top and taking the max value")
        return max_x, max_x - desired_shape_extent


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


def select_file(scan_name):
    folder_path = os.path.join(MRI_IMG_DIR, scan_name, '*', '*')
    all_files = glob.glob(folder_path)
    removal_vals = [x for x in all_files for skip_tag in filename_filter_keys if skip_tag in x]
    selected_folder = [x for x in all_files if x not in removal_vals]
    if len(selected_folder) != 1:
        bad_files.append(scan_name)
        raise AttributeError("Multiple candidate files. Please check scan_type criterion.")
    selected_folder = selected_folder[0]
    dicom_location = os.path.join(folder_path, selected_folder)
    dicom_files = os.listdir(os.path.join(folder_path, selected_folder))
    sorted(dicom_files, key=lambda x: x[2: x.find(".dcm")])
    scan = read_dicoms_as_np_array(dicom_files=dicom_files, dicom_location=dicom_location)
    return scan


def process_single_scan(scan_name, triage=True):
    row_begin, row_end, col_begin, col_end, slice_begin, slice_end = df.loc[scan_name]
    row_begin, row_end, col_begin, col_end, slice_begin, slice_end = obtain_tumor_dims(start_row=row_begin,
                                                                                       end_row=row_end,
                                                                                       start_col=col_begin,
                                                                                       end_col=col_end,
                                                                                       start_slice=slice_begin,
                                                                                       end_slice=slice_end)

    mri_scan = select_file(scan_name=scan_name)
    cropped_scan = mri_scan[slice_begin:slice_end, row_begin:row_end, col_begin:col_end]
    if triage:
        print(cropped_scan.shape)
        return
    np.save(os.path.join(SAVE_PATH, scan_name), cropped_scan)


def process_all_scans():
    for scan in tqdm(df.index):
        try:
            process_single_scan(scan_name=scan, triage=False)
        except AttributeError as e:
            print(f"Scan :- {scan} is bad!!!")
        print("continuing")


def handle_bad_files_from_first_round():
    bad_files = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'rb'))
    for scan in tqdm(bad_files):
        try:
            process_single_scan(scan_name=scan, triage=False)
        except AttributeError as e:
            print(f"Scan :- {scan} is bad!!!")


def sanity_check(train_split, val_split, test_split):
    train_set, val_set, test_set = set(train_split), set(val_split), set(test_split)
    length_arr = [len(train_set.intersection(val_set)), len(train_set.intersection(test_set)), len(test_set.intersection(val_set))]
    return all([x == 0 for x in length_arr])


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
    print(f"Creating dataset with train_len {len(train_split)}, val_len {len(val_split)} and test_len {len(test_split)}")
    pickle.dump(train_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'train.pkl'), 'wb'))
    pickle.dump(val_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'val.pkl'), 'wb'))
    pickle.dump(test_split, open(os.path.join(SPLIT_SAVE_FILE_PATH, 'test.pkl'), 'wb'))



if __name__ == '__main__':
    # tumor_data_file = os.path.join(META_DATA_DIR, 'Clinical_and_Other_Features.xlsx')
    # read_file(tumor_data_file)
    # process_all_scans()
    # process_single_scan(scan_name='Breast_MRI_001', triage=False)
    # pickle.dump(bad_files, open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'wb'))
    # handle_bad_files_from_first_round()
    # pickle.dump(bad_files, open(os.path.join(PROJECT_ROOT_DIR, 'bad_files.pkl'), 'wb'))
    train_val_test_splits()