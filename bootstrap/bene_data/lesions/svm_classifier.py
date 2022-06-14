import os

import numpy as np
from sklearn import svm

from ablation.radiomics_k_fold import evaluate_results
from bootstrap.utils.classical_models import execute_models
from environment_setup import PROJECT_ROOT_DIR

root_dir = os.path.join(PROJECT_ROOT_DIR, 'lesion', 'ssl_features_dir', 'lesion_ssl')


def classification(train_features, train_label, test_features):
    results = execute_models(train_features, train_label, test_features, 'svm')
    for method, preds in results.items():
        return preds
    # clf = svm.SVC(gamma='auto', C=1, class_weight='balanced', probability=True, kernel='linear', random_state=42, verbose=4)
    # clf.fit(train_features, train_label)
    # pred = clf.predict_proba(test_features)
    # return pred


def svm_training():
    train_features = np.load(os.path.join(root_dir, 'train_ssl_features.npy'))
    test_features = np.load(os.path.join(root_dir, 'test_ssl_features.npy'))
    train_labels = np.load(os.path.join(root_dir, 'train_ssl_labels.npy'))
    test_labels = np.load(os.path.join(root_dir, 'test_ssl_labels.npy'))
    print(f"Number of train samples: {train_features.shape[0]} and test samples: {test_features.shape[0]}")
    radiomics_pred = classification(train_features=train_features, train_label=train_labels,
                                    test_features=test_features)
    radiomics_pred = radiomics_pred[:, 1]
    specificity, sensitivity, auroc_value = evaluate_results(radiomics_pred=radiomics_pred, test_labels=test_labels)
    print(f"Sepcificity is {specificity}, Sensitivity is {sensitivity} and roc value is {auroc_value}")


if __name__ == '__main__':
    svm_training()
