import os

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder

from environment_setup import PROJECT_ROOT_DIR

base_dir = '/mnt/cat/chinmay/brats_processed'
radiomics_path = os.path.join(base_dir, 'data', 'radiomics_features', 'features_flair.npy')
labels_path = os.path.join(base_dir, 'label_all.npy')


def create_tsne_plot(features, labels, plot_name):
    # labels = enc.transform(labels.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # , learning_rate='auto',
    X_embedded = TSNE(n_components=2, init='random', verbose=True).fit_transform(features)
    target_ids = range(2)  # Since binary classification
    target_names = ['low', 'high']
    plt.figure(figsize=(6, 5))
    plt.figtext = plot_name
    colors = 'r', 'g'
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_embedded[labels == i, 0], X_embedded[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


def get_per_feat(radiomics):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps')
    test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
    test_X_combined_radiomics = np.concatenate((radiomics, test_X_vit), axis=-1)
    return test_X_combined_radiomics


def get_per_contrast_feat(radiomics):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps_contrast')
    test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_contrast_ssl_features.npy'))
    test_X_combined_radiomics = np.concatenate((radiomics, test_X_vit), axis=-1)
    return test_X_combined_radiomics


def plot_figs():
    radiomics, labels = get_radiomics_feat()
    create_tsne_plot(features=radiomics, labels=labels, plot_name='radiomics')
    # Now we go for perceptual and radiomics combined
    perc_features = get_per_feat(radiomics)
    create_tsne_plot(features=perc_features, labels=labels, plot_name='perc+radiomics')
    # Now we go for perceptual, contrastive and radiomics combined
    perc_contrast_features = get_per_contrast_feat(radiomics)
    create_tsne_plot(features=perc_contrast_features, labels=labels, plot_name='perc+contrast+radiomics')


def get_radiomics_feat():
    radiomics = np.asarray(np.load(radiomics_path))
    labels = np.load(labels_path)
    # train_idx = np.load(os.path.join(base_dir, 'data', 'train_indices.npy'))
    test_idx = np.load(os.path.join(base_dir, 'data', 'test_indices.npy'))
    test_X_rad_, test_y_ = radiomics[test_idx], labels[test_idx]
    return test_X_rad_, test_y_

if __name__ == '__main__':
    plot_figs()