import os

import pandas as pd
from sklearn.model_selection import train_test_split

from environment_setup import PROJECT_ROOT_DIR

csv_data_path = '/mnt/cat/chinmay/ms_lesions_graph/lesion_patches_gnn_updated.csv'
SAVE_PATH = os.path.join(PROJECT_ROOT_DIR, 'dataset', 'ms_lesion_dataset')


def split_data():
    data_csv = pd.read_csv(csv_data_path)
    train, val = train_test_split(data_csv)
    train.to_csv(os.path.join(SAVE_PATH, 'ssl.csv'), index=False)
    val.to_csv(os.path.join(SAVE_PATH, 'test.csv'), index=False)
    # Also saving the whole csv file in case we want to perform k-fold
    data_csv.to_csv(os.path.join(SAVE_PATH, 'whole.csv'), index=False)


if __name__ == '__main__':
    split_data()
