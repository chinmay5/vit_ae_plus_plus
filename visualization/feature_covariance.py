import os

from environment_setup import PROJECT_ROOT_DIR


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

feature_dir = os.path.join(PROJECT_ROOT_DIR, 'visualization', 'cov_dir')
split_index_path = os.path.join(PROJECT_ROOT_DIR, 'brats', 'k_fold', 'indices_file')

import seaborn as sns

import matplotlib.pyplot as plt


def generate_plots():
    breast_cancer = load_breast_cancer()
    data = breast_cancer.data
    features = breast_cancer.feature_names
    df = pd.DataFrame(data, columns=features)
    print(df.shape)
    print(features)
    # Including the radiomics features
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "egd_dataset")
    radiomics_test_features = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    radiomics_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))

    data_1 = read_perceptual_feat()
    print(np.shape(data_1))
    data_2 = read_contrastive_feat()
    print(np.shape(data_2))

    # data_1 = np.concatenate((radiomics_test_features, data_1), axis=1)
    # data_2 = np.concatenate((radiomics_test_features, data_2), axis=1)

    # print("Normalizing features")
    # normalize_features(features=data_1)
    # normalize_features(features=data_2)

    names = []
    for count, ii in enumerate(range(np.shape(data_1)[1])):
        names.append(str(count + 1))

    df = pd.DataFrame(data_1, columns=names)
    # df = df.iloc[:, :128]
    correlation_mat_1 = df.corr()
    sns.heatmap(correlation_mat_1, annot=False, cmap="YlGnBu")
    plt.show()

    # Creating histogram
    # n_bins = 20
    # fig, ax = plt.subplots(figsize=(10, 7))
    # ax.hist(correlation_mat_1, bins=n_bins, color=None)

    df = pd.DataFrame(data_2, columns=names)
    # df = df.iloc[:, :256]
    correlation_mat_2 = df.corr()
    sns.heatmap(correlation_mat_2, annot=False, cmap="YlGnBu")

    plt.show()

    df_1 = pd.DataFrame(correlation_mat_1, columns=names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, dpi=250)
    sns.distplot(correlation_mat_1, color="dodgerblue", ax=axes[0], axlabel='Covariance of Perceptual Representations')
    df_1 = pd.DataFrame(correlation_mat_2, columns=names)
    # fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True, dpi=250)
    sns.distplot(correlation_mat_2, color="deeppink", ax=axes[1],
                 axlabel='Covariance of Representations After Contrastive Training')

    plt.show()


def read_contrastive_feat(FILENAME='large_brats_one_stage'):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'egd_dataset', 'ssl_features_dir', FILENAME)
    # train_features = np.load(os.path.join(ssl_feature_dir, f'train_contrast_ssl_features_split_{0}.npy'))
    # test_features = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_features_split_{0}.npy'))
    test_features = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_features.npy'))

    # I found that normalizing features hurt performance in this case
    return test_features


def read_perceptual_feat(FILENAME='large_brats_ssl'):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'egd_dataset', 'ssl_features_dir', FILENAME)
    test_features = np.load(os.path.join(ssl_feature_dir, f'test_ssl_features.npy'))
    return test_features


# plt.show()


# df_2 = pd.DataFrame(correlation_mat_2, columns = names)
# #fig, axes = plt.subplots(2, 2, figsize=(10, 3), sharey=True, dpi=250)
# sns.distplot(df_2 , color="deeppink", ax=axes[0], axlabel='After KMS')


# correlation_mat_2 = df.corr()
# df_2 = pd.DataFrame(correlation_mat_2, columns = names)
# fig, axes = plt.subplots(2, 2, figsize=(10, 3), sharey=True, dpi=200)
# sns.distplot(df_1 , color="deeppink", ax=axes[0], axlabel='Before KMS')
#
# # Show plot
# # plt.show()
# #
# # df = pd.DataFrame(data_2, columns = names)
# # df = df.iloc[:,:64]
# # correlation_mat_2 = df.corr()
# # sns.heatmap(correlation_mat_2, annot = False, cmap="YlGnBu")
# #
# # plt.show()
#
#
#
#
# n_bins = 20
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.hist(correlation_mat_2, bins=n_bins)
#
# # Show plot
# plt.show()

if __name__ == '__main__':
    generate_plots()
