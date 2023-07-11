import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR

FILENAME = 'k_fold_egd'


def classification(train_features, train_label, test_features):
    results = execute_models(train_features, train_label, test_features, 'svm')  # 'svm') #, 'rf', 'linear')
    # results = execute_models(train_features, train_label, test_features, 'rf')  # 'svm') #, 'rf', 'linear')
    for method, preds in results.items():
        return preds
    # return pred


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector


def z_score_normalize(vector):
    vector -= np.mean(vector)
    vector = vector / (2 * np.std(vector) + 0.001)
    return vector


def normalize_features(features):
    for ii in range(np.shape(features)[1]):
        #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
        features[:, ii] = min_max_normalize(features[:, ii], 1)


def evaluate_results(radiomics_pred, test_labels):
    auroc_value = roc_auc_score(test_labels, radiomics_pred)
    radiomics_pred[radiomics_pred >= 0.65] = 1
    radiomics_pred[radiomics_pred < 0.65] = 0
    cm = confusion_matrix(radiomics_pred, test_labels)
    # print(cm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('specificity:', specificity)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    return specificity, sensitivity, auroc_value


def work_on_ssl_features(idx):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'egd', 'ssl_features_dir', FILENAME)
    train_features = np.load(os.path.join(ssl_feature_dir, f'train_contrast_ssl_features_split_{idx}.npy'))
    test_features = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_features_split_{idx}.npy'))
    train_labels = np.load(os.path.join(ssl_feature_dir, f'train_contrast_ssl_labels_split_{idx}.npy'))
    test_labels = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_labels_split_{idx}.npy'))
    # I found that normalizing features hurt performance in this case
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, auroc_value = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, auroc_value


def evaluate_features():
    # At this moment, the evaluation is for Brats only.
    split_index_path = os.path.join(PROJECT_ROOT_DIR, 'egd', 'k_fold', 'indices_file')
    avg_auroc_value_ssl = []
    avg_sensitivity_ssl, avg_specificity_ssl = [], []
    for idx in range(3):
        train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{idx}"), 'rb'))
        test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))

        specificity_ssl_contr, sensitivity_ssl_contr, auroc_value_ssl_contr = work_on_ssl_features(idx=idx)
        avg_specificity_ssl.append(specificity_ssl_contr)
        avg_sensitivity_ssl.append(sensitivity_ssl_contr)
        # Including the auroc values
        avg_auroc_value_ssl.append(auroc_value_ssl_contr)
    print("SSL Features")
    print(
        f"Average specificity {np.mean(avg_specificity_ssl)} and sensitivity {np.mean(avg_sensitivity_ssl)} roc: {np.mean(avg_auroc_value_ssl)}")
    print(f"roc std: {np.std(avg_auroc_value_ssl)}")


if __name__ == '__main__':
    evaluate_features()
