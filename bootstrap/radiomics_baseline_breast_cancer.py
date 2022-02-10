import numpy as np
import os

from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import confusion_matrix, roc_auc_score


def read_csv(name):
    features = []
    with open(name) as f:
        lis = [line.split() for line in f]  # create a list of lists
        for i, x in enumerate(lis):  # print the list items
            #  print ("line{0} = {1}".format(i, x))
            features.append(x[1])
    return features[30:110]


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector


def z_score_normalize(vector):
    vector -= np.mean(vector)
    vector = vector / (2 * np.std(vector) + 0.001)
    return vector


BASE_DIR = os.path.join('/mnt', 'cat', 'chinmay', 'duke_breast_cancer')
RADIOMICS_SAVE_FILE_PATH = os.path.join(BASE_DIR, 'mri_images', 'Duke-Breast-Cancer-MRI_v120201203', 'radiomics_feat')

base_dir = '/mnt/cat/chinmay/brats_processed'


def bootstrap(option='radiomics', subtype=None):
    def normalize_features(numpy_arr):
        for ii in range(np.shape(numpy_arr)[1]):
            #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
            numpy_arr[:, ii] = min_max_normalize(numpy_arr[:, ii], 1)

    if option == 'radiomics':
        train_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_feat.npy'))
        train_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_labels.npy'))
        test_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_feat.npy'))
        test_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'))

    elif option == 'ssl':
        assert subtype is not None, "Please specify the subtype: perc, contrast etc"
        ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'breast_cancer', 'ssl_features_dir', subtype)  # Add subtype later on
        train_numpy_feat = np.load(os.path.join(ssl_feature_dir, 'features.npy'))
        test_numpy_feat = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
        train_numpy_labels = np.load(os.path.join(ssl_feature_dir, 'gt_labels.npy'))
        test_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'))
    elif option == 'combined':
        assert subtype is not None, "Please specify the subtype: perc, contrast etc"
        ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'breast_cancer', 'ssl_features_dir', subtype)  # Add subtype later on
        train_numpy_feat_ssl = np.load(os.path.join(ssl_feature_dir, 'features.npy'))
        test_numpy_feat_ssl = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
        train_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_feat.npy'))
        train_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_labels.npy'))
        test_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_feat.npy'))
        test_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'))
        train_numpy_feat = np.concatenate([train_numpy_feat, train_numpy_feat_ssl], axis=1)
        test_numpy_feat = np.concatenate([test_numpy_feat, test_numpy_feat_ssl], axis=1)

    else:
        raise AttributeError(f"Invalid option {option} used")


    normalize_features(train_numpy_feat)
    normalize_features(test_numpy_feat)

    print(f"Number of train samples: {train_numpy_feat.shape[0]} and test samples: {test_numpy_feat.shape[0]}")
    return train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels


def evaluate_results(pred, label):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    cm = confusion_matrix(pred, label)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    return specificity, sensitivity, cm

def evaluate_models(train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels):
    results = execute_models(train_numpy_feat, train_numpy_labels, test_numpy_feat, 'svm', 'rf', 'linear')
    for method, preds in results.items():
        preds = preds[:, 1]
        specificity, sensitivity, cm = evaluate_results(pred=preds, label=test_numpy_labels)
        print(f"Method: {method}, \n Specificity: {specificity}, \n Sensitivity: {sensitivity}\n {cm}")


#############################################################################################################

if __name__ == '__main__':
    print("---------RADIOMICS ALONE---------")
    train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels = bootstrap(option='radiomics')
    evaluate_models(train_numpy_feat=train_numpy_feat, train_numpy_labels=train_numpy_labels,
                    test_numpy_feat=test_numpy_feat, test_numpy_labels=test_numpy_labels)

    print("---------SSL ALONE---------")
    train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels = bootstrap(option='ssl', subtype='contrast')
    evaluate_models(train_numpy_feat=train_numpy_feat, train_numpy_labels=train_numpy_labels,
                    test_numpy_feat=test_numpy_feat, test_numpy_labels=test_numpy_labels)

    print("---------COMBINED FEATURES---------")
    train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels = bootstrap(option='combined', subtype='contrast')
    evaluate_models(train_numpy_feat=train_numpy_feat, train_numpy_labels=train_numpy_labels,
                    test_numpy_feat=test_numpy_feat, test_numpy_labels=test_numpy_labels)
