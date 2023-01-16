import os
import pickle

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR

FILENAME = 'egd_percep_all'


def classification(train_features, train_label, test_features, is_50=False):
    if is_50:
        train_features, _, train_label, _ = train_test_split(train_features, train_label, test_size=0.50,
                                                             random_state=42)
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


def work_on_radiomics_features(train_ids, test_ids):
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "egd")
    radiomic_features_mapped = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    radiomics_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))
    radiomics_train_feat, train_labels = radiomic_features_mapped[train_ids], radiomics_labels[train_ids]
    radiomics_test_features, test_labels = radiomic_features_mapped[test_ids], radiomics_labels[test_ids]
    # noramlize the features
    normalize_features(features=radiomics_train_feat)
    normalize_features(features=radiomics_test_features)
    radiomics_pred = classification(train_features=radiomics_train_feat, train_label=train_labels,
                                    test_features=radiomics_test_features)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, auroc_value = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, auroc_value


def work_on_contrast_features(idx):
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


def work_on_contrast_combined_features(train_ids, test_ids, idx):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'egd', 'ssl_features_dir', FILENAME)
    train_features = np.load(os.path.join(ssl_feature_dir, f'train_contrast_ssl_features_split_{idx}.npy'))
    test_features = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_features_split_{idx}.npy'))
    ssl_train_labels = np.load(os.path.join(ssl_feature_dir, f'train_contrast_ssl_labels_split_{idx}.npy'))
    ssl_test_labels = np.load(os.path.join(ssl_feature_dir, f'test_contrast_ssl_labels_split_{idx}.npy'))
    # Now the radiomic features
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "egd")
    radiomic_features_mapped = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    radiomics_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))
    radiomics_train_feat, train_labels = radiomic_features_mapped[train_ids], radiomics_labels[train_ids]
    radiomics_test_features, test_labels = radiomic_features_mapped[test_ids], radiomics_labels[test_ids]
    # Concatenate the features
    assert np.all(train_labels == ssl_train_labels), "Something wrong with train labels"
    assert np.all(test_labels == ssl_test_labels), "Something wrong with test labels"
    train_features = np.concatenate((radiomics_train_feat, train_features), axis=1)
    test_features = np.concatenate((radiomics_test_features, test_features), axis=1)
    # noramlize the features
    normalize_features(features=train_features)
    normalize_features(features=test_features)
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, auroc_value = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, auroc_value


def evaluate_features():
    # At this moment, the evaluation is for Brats only.
    split_index_path = os.path.join(PROJECT_ROOT_DIR, 'egd', 'k_fold', 'indices_file')
    avg_auroc_value, avg_auroc_value_ssl_contr, avg_auroc_value_ssl_comb_contr = 0, 0, 0
    avg_specificity, avg_sensitivity, avg_sensitivity_ssl_contr, avg_specificity_ssl_contr, avg_specificity_comb_contr, avg_sensitivity_comb_contr = 0, 0, 0, 0, 0, 0
    for idx in range(3):
        train_ids = pickle.load(open(os.path.join(split_index_path, f"train_{idx}"), 'rb'))
        test_ids = pickle.load(open(os.path.join(split_index_path, f"test_{idx}"), 'rb'))
        specificity, sensitivity, auroc_value = work_on_radiomics_features(train_ids=train_ids, test_ids=test_ids)

        specificity_ssl_contr, sensitivity_ssl_contr, auroc_value_ssl_contr = work_on_contrast_features(idx=idx)
        specificity_ssl_comb_contr, sensitivity_ssl_comb_contr, auroc_value_ssl_comb_contr = work_on_contrast_combined_features(
            train_ids=train_ids, test_ids=test_ids, idx=idx)
        avg_sensitivity += sensitivity
        avg_specificity += specificity
        avg_specificity_ssl_contr += specificity_ssl_contr
        avg_sensitivity_ssl_contr += sensitivity_ssl_contr
        avg_specificity_comb_contr += specificity_ssl_comb_contr
        avg_sensitivity_comb_contr += sensitivity_ssl_comb_contr
        # Including the auroc values
        avg_auroc_value += auroc_value
        avg_auroc_value_ssl_contr += auroc_value_ssl_contr
        avg_auroc_value_ssl_comb_contr += auroc_value_ssl_comb_contr
    print("Radiomics features")
    print(
        f"Average specificity {avg_specificity / 3} and sensitivity {avg_sensitivity / 3}, roc: {avg_auroc_value / 3}")
    print("Contrast Features")
    print(
        f"Average specificity {avg_specificity_ssl_contr / 3} and sensitivity {avg_sensitivity_ssl_contr / 3} roc: {avg_auroc_value_ssl_contr / 3}")
    print("Combined Features")
    print(
        f"Average specificity {avg_specificity_comb_contr / 3} and sensitivity {avg_sensitivity_comb_contr / 3} roc: {avg_auroc_value_ssl_comb_contr / 3}")


if __name__ == '__main__':
    evaluate_features()
