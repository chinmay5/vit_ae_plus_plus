from collections import defaultdict

import numpy as np
import os

from sklearn.model_selection import KFold, StratifiedKFold

from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


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


def bootstrap(option='radiomics', subtype=None, filename="clinical"):
    def normalize_features(numpy_arr):
        for ii in range(np.shape(numpy_arr)[1]):
            #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
            numpy_arr[:, ii] = min_max_normalize(numpy_arr[:, ii], 1)

    if option == 'radiomics':
        features = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, f'{filename}_features_all.npy'))
        labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, f'{filename}_labels_all.npy'))

    elif option == 'ssl':
        assert subtype is not None, "Please specify the subtype: perc, contrast etc"
        ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, '', 'ssl_features_dir', subtype)  # Add subtype later on
        train_numpy_feat = np.load(os.path.join(ssl_feature_dir, 'features.npy'))
        test_numpy_feat = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
        train_numpy_labels = np.load(os.path.join(ssl_feature_dir, 'gt_labels.npy'))
        test_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'))
    elif option == 'combined':
        assert subtype is not None, "Please specify the subtype: perc, contrast etc"
        ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, '', 'ssl_features_dir', subtype)  # Add subtype later on
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


    normalize_features(features)

    print(f"Number of train samples: {features.shape[0]} and test samples: {labels.shape[0]}")
    return features, labels


def evaluate_results(pred, label, target_names=['0', '1', '2']):
    pred = np.argmax(pred, axis=1)
    print(confusion_matrix(label, pred)) #(label, pred, target_names=target_names))
    # return specificity, sensitivity, cm


def process_k_fold_results(per_model_result_dict):
    formatted_dict = {}
    for method, spec_sen_cm_list in per_model_result_dict.items():
        spec_all, sen_all, cm_all, cnt = 0, 0, 0, 1
        for spec, sen, cm in spec_sen_cm_list:
            spec_all += spec
            sen_all += sen
            cm_all += cm
            cnt += 1
        # formatted_dict[method] = (spec_all / cnt, sen_all/cnt, cm_all/cnt)
        formatted_dict[method] = (spec_all / cnt, sen_all/cnt)
    print(formatted_dict)



def evaluate_models(features, labels):
    kfold_splits = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
    per_model_pred_dict = defaultdict(list)
    per_model_label_dict = defaultdict(list)
    for train_index, test_index in kfold_splits.split(features, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        results = execute_models(X_train, y_train, X_test, 'svm', 'rf', 'linear')
        for method, preds in results.items():
            per_model_pred_dict[method].append(preds)
            per_model_label_dict[method].append(y_test)
    # Convert individual ml-model predictions to numpy array for easy handling
    for method, pred_list in per_model_pred_dict.items():
        # TODO: Perhaps make a class that stores pred and label so that we need not do it in this "hacky" way
        pred_arr = np.concatenate(pred_list)
        label_arr = np.concatenate(per_model_label_dict[method])
        print(f"Results for {method}")
        evaluate_results(pred=pred_arr, label=label_arr)
        # print(f"Method: {method}, \n Specificity: {specificity}, \n Sensitivity: {sensitivity}\n {cm}")


#############################################################################################################

if __name__ == '__main__':
    print("---------RADIOMICS ALONE---------")
    features, labels = bootstrap(option='radiomics', filename='clinical')
    evaluate_models(features=features, labels=labels)

    # features, labels = bootstrap(option='radiomics', filename='pathology')
    # evaluate_models(features=features, labels=labels)

    # print("---------SSL ALONE---------")
    # train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels = bootstrap(option='ssl', subtype='contrast_no_l2')
    # evaluate_models(train_numpy_feat=train_numpy_feat, train_numpy_labels=train_numpy_labels,
    #                 test_numpy_feat=test_numpy_feat, test_numpy_labels=test_numpy_labels)
    #
    # print("---------COMBINED FEATURES---------")
    # train_numpy_feat, train_numpy_labels, test_numpy_feat, test_numpy_labels = bootstrap(option='combined', subtype='contrast_no_l2')
    # evaluate_models(train_numpy_feat=train_numpy_feat, train_numpy_labels=train_numpy_labels,
    #                 test_numpy_feat=test_numpy_feat, test_numpy_labels=test_numpy_labels)
