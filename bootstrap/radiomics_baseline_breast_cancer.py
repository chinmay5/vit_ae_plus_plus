import numpy as np
from sklearn import svm, datasets
import os
from sklearn import metrics

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

train_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_feat.npy'))
train_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'train_labels.npy'))
val_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'val_feat.npy'))
val_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'val_labels.npy'))
test_numpy_feat = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_feat.npy'))
test_numpy_labels = np.load(os.path.join(RADIOMICS_SAVE_FILE_PATH, 'test_labels.npy'))

epochs = 45


def normalize_features(numpy_arr):
    for ii in range(np.shape(numpy_arr)[1]):
        #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
        numpy_arr[:, ii] = min_max_normalize(numpy_arr[:, ii], 1)


normalize_features(train_numpy_feat)
normalize_features(val_numpy_feat)
normalize_features(test_numpy_feat)

# We need to also convert the classification task into

def classification(train_features, train_label, test_features):
    clf = svm.SVC(gamma='auto', C=1, class_weight='balanced', probability=True, kernel='linear', random_state=42)
    clf.fit(train_features, train_label)
    pred = clf.predict_proba(test_features)
    return pred


temp_pred_rad = classification(train_features=train_numpy_feat, train_label=train_numpy_labels,
                               test_features=test_numpy_feat)

print(f"Number of train samples: {train_numpy_feat.shape[0]} and test samples: {test_numpy_feat.shape[0]}")

temp_label = test_numpy_labels
temp_pred_rad = temp_pred_rad[:, 1]

# fig = plt.figure(1)
# plot = fig.add_subplot(111)
# fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_rad)
# auc = metrics.roc_auc_score(temp_label, temp_pred_rad)
# plt.plot(fpr,tpr,label="trad. radiomics, AUC = "+str(auc)[0:5])
# plt.legend(loc=4, prop={'size': 12})

temp_pred_rad_ = temp_pred_rad
temp_pred_rad[temp_pred_rad >= 0.5] = 1
temp_pred_rad[temp_pred_rad < 0.5] = 0
cm = confusion_matrix(temp_pred_rad, temp_label)
print(cm)
specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
print('Radiomics alone:')
print('specificity:', specificity)
sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
print('sensitivity:', sensitivity)
