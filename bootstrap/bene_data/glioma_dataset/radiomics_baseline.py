import os
import pickle

import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR


def classification(train_features, train_label, test_features, is_50=False):
    # clf = svm.SVC(gamma='auto', C=1, class_weight='balanced', probability=True, kernel='linear', random_state=42)
    # clf.fit(train_features, train_label)
    # pred = clf.predict_proba(test_features)
    # return pred
    # results = execute_models(train_features, train_label, test_features, 'svm')  # 'svm') #, 'rf', 'linear')
    if is_50:
        train_features, _, train_label, _ = train_test_split(train_features, train_label, test_size=0.50,
                                                             random_state=42, stratify=train_label)
    results = execute_models(train_features, train_label, test_features, 'svm')  # 'svm') #, 'rf', 'linear')
    for method, preds in results.items():
        return preds


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector
def z_score_normalize(vector):
    vector -= np.mean(vector)
    vector = vector / (2*np.std(vector)+0.001)
    return vector

def normalize_features(features):
    for ii in range(np.shape(features)[1]):
    #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
        features[:, ii] = min_max_normalize(features[:, ii], 1)

def evaluate_results(radiomics_pred, test_labels):
    roc_score = roc_auc_score(test_labels, radiomics_pred)
    radiomics_pred[radiomics_pred >= 0.50] = 1
    radiomics_pred[radiomics_pred < 0.50] = 0
    cm = confusion_matrix(radiomics_pred, test_labels)
    # print(cm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('specificity:', specificity)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    return specificity, sensitivity, roc_score


def work_on_radiomics_features(train_ids, test_ids, is_50=False):
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "large_brats")
    radiomic_features_mapped = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    radiomics_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))

    radiomics_train_feat, train_labels = radiomic_features_mapped[train_ids], radiomics_labels[train_ids]
    radiomics_test_features, test_labels = radiomic_features_mapped[test_ids], radiomics_labels[test_ids]
    # noramlize the features
    normalize_features(features=radiomics_train_feat)
    normalize_features(features=radiomics_test_features)
    radiomics_pred = classification(train_features=radiomics_train_feat, train_label=train_labels,
                                    test_features=radiomics_test_features, is_50=is_50)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, roc_score = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, roc_score


def map_radiomic_features():
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "large_brats")
    downstream_scans = pickle.load(
        open(os.path.join(SPLIT_SAVE_FILE_PATH, 'who_idh_mutation_status_annotated_mit_labels.pkl'), 'rb'))
    # Now we need to load the radiomics feature file.
    radiomics_dir = '/mnt/cat/chinmay/glioma_Bene'
    radiomics_features = np.load(os.path.join(radiomics_dir, 'radiomics_all.npy'))
    pat_list = np.load(os.path.join(radiomics_dir, 'pat_lists.npy'))
    all_features, all_labels = [], []
    # Get the index
    for mri_scan, label in downstream_scans:
        index = np.where(pat_list == mri_scan)
        all_features.append(radiomics_features[index])
        all_labels.append(label)
    radiomic_features_mapped = np.concatenate(all_features)
    radiomics_labels = np.stack(all_labels)
    # Somehow, the radiomics features were stored as strings and thus, we need to cast them to float values.
    radiomic_features_mapped = radiomic_features_mapped.astype(np.float)
    np.save(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped'), radiomic_features_mapped)
    np.save(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped'), radiomics_labels)


def work_on_ssl_features(train_ids, test_ids, filename='large_brats_ssl', is_50=False):
    # filename = 'large_brats_no_perceptual'
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', filename)
    features = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
    labels = np.load(os.path.join(ssl_feature_dir, 'test_ssl_labels.npy'))

    train_features, test_features = features[train_ids], features[test_ids]
    train_labels, test_labels = labels[train_ids], labels[test_ids]
    # Normalizing SSL features hurt performance. Still, double-check
    # normalize_features(features=train_features)
    # normalize_features(features=test_features)

    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features, is_50=is_50)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, roc_score = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, roc_score


def work_on_contrast_ssl_features(train_ids, test_ids,is_50=False, filename='all_comps_large_brats'):
    # filename = 'large_brats_no_perceptual'
    contr_ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', filename)
    features = np.load(os.path.join(contr_ssl_feature_dir, 'test_contrast_ssl_features.npy'))
    labels = np.load(os.path.join(contr_ssl_feature_dir, 'test_contrast_ssl_labels.npy'))
    # Normalizing SSL features hurt performance. Still, double-check
    # normalize_features(features=train_features)
    # normalize_features(features=test_features)
    train_features, test_features = features[train_ids], features[test_ids]
    train_labels, test_labels = labels[train_ids], labels[test_ids]
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features,is_50=is_50)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, roc_score = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, roc_score



def work_on_combined_contr_features(train_ids, test_ids, filename='all_comps_large_brats',is_50=False):
    # filename = 'large_brats_no_perceptual'
    contr_ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', filename)
    features = np.load(os.path.join(contr_ssl_feature_dir, 'test_contrast_ssl_features.npy'))
    labels = np.load(os.path.join(contr_ssl_feature_dir, 'test_contrast_ssl_labels.npy'))
    train_features, test_features = features[train_ids], features[test_ids]
    ssl_train_labels, ssl_test_labels = labels[train_ids], labels[test_ids]
    # Include the radiomics features
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "large_brats")
    radiomics_features = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    all_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))

    radiomics_train_feat, train_labels = radiomics_features[train_ids], all_labels[train_ids]
    radiomics_test_features, test_labels = radiomics_features[test_ids], all_labels[test_ids]
    # Concatenate the features
    assert np.all(train_labels == ssl_train_labels), "Something wrong with train labels"
    assert np.all(test_labels == ssl_test_labels), "Something wrong with test labels"
    train_features = np.concatenate((radiomics_train_feat, train_features), axis=1)
    test_features = np.concatenate((radiomics_test_features, test_features), axis=1)
    # noramlize the features
    normalize_features(features=train_features)
    normalize_features(features=test_features)
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features,is_50=is_50)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, roc_score = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, roc_score

def work_on_combined_features(train_ids, test_ids, filename='large_brats_ssl',is_50=False):
    # filename = 'large_brats_no_perceptual'
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', filename)
    features = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
    labels = np.load(os.path.join(ssl_feature_dir, 'test_ssl_labels.npy'))
    train_features, test_features = features[train_ids], features[test_ids]
    ssl_train_labels, ssl_test_labels = labels[train_ids], labels[test_ids]
    # Include the radiomics features
    SPLIT_SAVE_FILE_PATH = os.path.join(PROJECT_ROOT_DIR, "dataset", "large_brats")
    radiomics_features = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomic_features_mapped.npy'))
    all_labels = np.load(os.path.join(SPLIT_SAVE_FILE_PATH, 'radiomics_labels_mapped.npy'))

    radiomics_train_feat, train_labels = radiomics_features[train_ids], all_labels[train_ids]
    radiomics_test_features, test_labels = radiomics_features[test_ids], all_labels[test_ids]
    # Concatenate the features
    assert np.all(train_labels == ssl_train_labels), "Something wrong with train labels"
    assert np.all(test_labels == ssl_test_labels), "Something wrong with test labels"
    train_features = np.concatenate((radiomics_train_feat, train_features), axis=1)
    test_features = np.concatenate((radiomics_test_features, test_features), axis=1)
    # noramlize the features
    normalize_features(features=train_features)
    normalize_features(features=test_features)
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features,is_50=is_50)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, roc_score = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    return specificity, sensitivity, roc_score

# TODO: Broken with filename

# def evaluate_features(is_50=False, filename='all_comps_large_brats'):
def evaluate_features(is_50=False, filename='large_brats_ssl'):
    ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', 'large_brats_ssl')
    # ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'large_brats', 'ssl_features_dir', 'large_brats_ssl')
    features = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
    labels = np.load(os.path.join(ssl_feature_dir, 'test_ssl_labels.npy'))
    avg_specificity, avg_sensitivity, avg_sensitivity_ssl, avg_specificity_ssl, avg_specificity_comb, avg_sensitivity_comb = 0, 0, 0, 0, 0, 0
    # avg_sensitivity_ssl_contr, avg_specificity_ssl_contr, avg_specificity_comb_contr, avg_sensitivity_comb_contr = 0, 0, 0, 0
    avg_roc_comb, avg_roc_ssl, avg_roc = 0, 0, 0
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    for train_ids, test_ids in skf.split(features, labels):
        specificity, sensitivity, roc_score = work_on_radiomics_features(train_ids=train_ids, test_ids=test_ids,is_50=is_50)
        specificity_ssl, sensitivity_ssl, roc_score_ssl = work_on_ssl_features(train_ids=train_ids, test_ids=test_ids,is_50=is_50, filename=filename)
        specificity_ssl_comb, sensitivity_ssl_comb, roc_score_comb = work_on_combined_features(train_ids=train_ids, test_ids=test_ids,is_50=is_50, filename=filename)
        # specificity_ssl_contr, sensitivity_ssl_contr = work_on_contrast_ssl_features(train_ids=train_ids, test_ids=test_ids,is_50=is_50, filename=filename)
        # specificity_ssl_comb_contr, sensitivity_ssl_comb_contr = work_on_combined_contr_features(train_ids=train_ids, test_ids=test_ids,is_50=is_50, filename=filename)
        avg_sensitivity += sensitivity
        avg_specificity += specificity
        avg_sensitivity_ssl += sensitivity_ssl
        avg_specificity_ssl += specificity_ssl
        avg_sensitivity_comb += sensitivity_ssl_comb
        avg_specificity_comb += specificity_ssl_comb
        avg_roc_comb += roc_score_comb
        avg_roc_ssl += roc_score_ssl
        avg_roc += roc_score
        # avg_specificity_ssl_contr += specificity_ssl_contr
        # avg_sensitivity_ssl_contr += sensitivity_ssl_contr
        # avg_specificity_comb_contr += specificity_ssl_comb_contr
        # avg_sensitivity_comb_contr += sensitivity_ssl_comb_contr
    print("Radiomics features")
    print(f"Average specificity {avg_specificity/n_splits} and sensitivity {avg_sensitivity/n_splits} roc {avg_roc/n_splits}")
    print("SSL Features")
    print(f"Average specificity {avg_specificity_ssl/n_splits} and sensitivity {avg_sensitivity_ssl/n_splits} roc {avg_roc_ssl/n_splits}")
    print("Combined Features")
    print(f"Average specificity {avg_specificity_comb / n_splits} and sensitivity {avg_sensitivity_comb / n_splits} roc {avg_roc_comb/n_splits}")
    # print("Contrast features")
    # print(f"Average specificity {avg_specificity_ssl_contr / n_splits} and sensitivity {avg_sensitivity_ssl_contr / n_splits}")
    # print("Contrast Combined Features")
    # print(
    #     f"Average specificity {avg_specificity_comb_contr / n_splits} and sensitivity {avg_sensitivity_comb_contr / n_splits}")
    return [avg_specificity/n_splits, avg_specificity_ssl/n_splits, avg_specificity_comb / n_splits],\
           [avg_sensitivity/n_splits, avg_sensitivity_ssl/n_splits, avg_sensitivity_comb / n_splits]



if __name__ == '__main__':
    # map_radiomic_features()
    evaluate_features(is_50=False)
    # evaluate_imagenet_features()
