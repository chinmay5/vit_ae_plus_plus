import os

import numpy as np
from sklearn import metrics
from sklearn import svm

from environment_setup import PROJECT_ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def read_csv(name):
    features = []
    with open(name) as f:
        lis = [line.split() for line in f]        # create a list of lists
        for i, x in enumerate(lis):              #print the list items
          #  print ("line{0} = {1}".format(i, x))
            features.append(x[1])
    return features[30:110]
def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector
def z_score_normalize(vector):
    vector -= np.mean(vector)
    vector = vector / (2*np.std(vector)+0.001)
    return vector

base_dir = '/mnt/cat/chinmay/brats_processed'
radiomics_path = os.path.join(base_dir, 'data', 'radiomics_features', 'features_flair.npy')
f1_path = os.path.join(base_dir, 'data', 'SS_features_45_orig_imbalanced.npy')
f2_path = os.path.join(base_dir, 'data', 'SS_features_k_means_k3_b6250.npy')
f3_path = os.path.join(base_dir, 'data', 'SS_features_k_means_k3_b61450_re-weighting.npy')
labels_path = os.path.join(base_dir, 'label_all.npy')

radiomics = np.asarray(np.load(radiomics_path))
epochs = 45

f = np.load(f1_path)
f_2 = np.load(f2_path)
f_3 = np.load(f3_path)

for ii in range(np.shape(radiomics)[1]):
#    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
    radiomics[:, ii] = min_max_normalize(radiomics[:, ii], 1)

for ii in range(np.shape(f)[1]):
    # f[:, ii] = z_score_normalize(f[:, ii])
    # f_2[:, ii] = z_score_normalize(f_2[:, ii])
    f[:, ii] = min_max_normalize(f[:, ii], 1)
    f_2[:, ii] = min_max_normalize(f_2[:, ii], 1)
    f_3[:, ii] = min_max_normalize(f_3[:, ii], 1)

# print(np.shape(radiomics))
labels = np.load(labels_path)

train_idx = np.load(os.path.join(base_dir, 'data', 'train_indices.npy'))
test_idx = np.load(os.path.join(base_dir, 'data', 'test_indices.npy'))

train_X_rad_, test_X_rad_ = radiomics[train_idx], radiomics[test_idx]
train_X_ssl_, test_X_ssl_ = f[train_idx], f[test_idx]
train_X_ssl_KMS_, test_X_ssl_KMS_ = f_2[train_idx], f_2[test_idx]
train_X_ssl_RW_, test_X_ssl_RW_ = f_3[train_idx], f_3[test_idx]

train_y_, test_y_ = labels[train_idx], labels[test_idx]

# Let us also save the values for training our SSL models and making everything deterministic
train_indices_save_path = os.path.join(base_dir, 'data', 'train_indices.npy')
test_indices_save_path = os.path.join(base_dir, 'data', 'test_indices.npy')
np.save(train_indices_save_path, train_idx)
np.save(test_indices_save_path, test_idx)

# skf = StratifiedKFold(n_splits=5)
# skf_50 = StratifiedKFold(n_splits=2)
#print(np.shape(skf))

def classification(train_features, train_label, test_features):
   # lin_clf = svm.LinearSVC()
   # lin_clf.fit(train_features, train_label)
    clf = svm.SVC(gamma='auto', C=1, class_weight='balanced', probability=True, kernel='linear',random_state=42)
    clf.fit(train_features, train_label)
    # svm.SVC(C=100.0, cache_size=200, class_weight=False, coef0=0.0,
    # decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    # max_iter=-1, probability=False, random_state=None, shrinking=True,
    # tol=0.001, verbose=False)
    #clf.probability = True
    pred = clf.predict_proba(test_features)
    #pred = clf.predict(test_features)
    return pred

train_X_combined_ = np.concatenate((train_X_rad_, train_X_ssl_), axis=-1)
train_X_combined_2_ = np.concatenate((train_X_rad_, train_X_ssl_KMS_), axis=-1)
train_X_combined_3_ = np.concatenate((train_X_rad_, train_X_ssl_RW_), axis=-1)



test_X_combined_ = np.concatenate((test_X_rad_, test_X_ssl_), axis=-1)
test_X_combined_2_ = np.concatenate((test_X_rad_, test_X_ssl_KMS_), axis=-1)
test_X_combined_3_ = np.concatenate((test_X_rad_, test_X_ssl_RW_), axis=-1)
#val_X_combined_3 = np.concatenate((val_X_rad, val_X_ssl_KMS_add), axis=-1)

temp_pred_rad = classification(train_features=train_X_rad_, train_label=train_y_, test_features=test_X_rad_)
temp_pred_ssl = classification(train_features=train_X_ssl_, train_label=train_y_, test_features=test_X_ssl_)
temp_pred_ssl_KMS = classification(train_features=train_X_ssl_KMS_, train_label=train_y_, test_features=test_X_ssl_KMS_)
temp_pred_ssl_RW = classification(train_features=train_X_ssl_RW_, train_label=train_y_, test_features=test_X_ssl_RW_)

temp_pred_combined = classification(train_features=train_X_combined_, train_label=train_y_, test_features=test_X_combined_)
temp_pred_combined_2 = classification(train_features=train_X_combined_2_, train_label=train_y_, test_features=test_X_combined_2_)
temp_pred_combined_3 = classification(train_features=train_X_combined_3_, train_label=train_y_, test_features=test_X_combined_3_)
#   temp_pred_combined_3 = classification(train_X_combined_3, train_y, val_X_combined_3)

print(f"Number of train samples: {train_X_combined_.shape} and test samples: {temp_pred_combined.shape}")

temp_label = test_y_
temp_pred_rad = temp_pred_rad[:, 1]
temp_pred_ssl = temp_pred_ssl[:, 1]
temp_pred_ssl_KMS = temp_pred_ssl_KMS[:, 1]
temp_pred_ssl_RW = temp_pred_ssl_RW[:, 1]
temp_pred_combined = temp_pred_combined[:, 1]
temp_pred_combined_2 = temp_pred_combined_2[:, 1]
temp_pred_combined_3 = temp_pred_combined_3[:, 1]


fig = plt.figure(1)
plot = fig.add_subplot(111)
# print(np.shape(temp_label))
# print(np.shape(temp_pred_rad))
fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_rad)
auc = metrics.roc_auc_score(temp_label, temp_pred_rad)
plt.plot(fpr,tpr,label="trad. radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_ssl)
auc = metrics.roc_auc_score(temp_label, temp_pred_ssl)
plt.plot(fpr,tpr,label="SSL radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_ssl_KMS)
auc = metrics.roc_auc_score(temp_label, temp_pred_ssl_KMS)
plt.plot(fpr,tpr,label="SSL+KMS radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_ssl_RW)
auc = metrics.roc_auc_score(temp_label, temp_pred_ssl_RW)
plt.plot(fpr,tpr,label="SSL+RW radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_combined)
auc = metrics.roc_auc_score(temp_label, temp_pred_combined)
plt.plot(fpr,tpr,label="trad. + SSL radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_combined_2)
auc = metrics.roc_auc_score(temp_label, temp_pred_combined_2)
plt.plot(fpr,tpr,label="trad. + SSL-KMS radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

plot.tick_params(axis='x', labelsize=13)
plot.tick_params(axis='y', labelsize=13)
plt.yticks(fontsize=13)

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_combined_3)
auc = metrics.roc_auc_score(temp_label, temp_pred_combined_3)
plt.plot(fpr,tpr,label="trad. + SSL-RW radiomics, AUC = "+str(auc)[0:5])
plt.legend(loc=4, prop={'size': 12})

plot.tick_params(axis='x', labelsize=13)
plot.tick_params(axis='y', labelsize=13)
plt.yticks(fontsize=13)

plt.show()

# fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_combined_2)
# auc = metrics.roc_auc_score(temp_label, temp_pred_combined_3)
# plt.plot(fpr,tpr,label="trad. + SSL-KMS' radiomics, auc="+str(auc)[0:5])
# plt.legend(loc=4, prop={'size': 12})
# plt.show()

# print(temp_pred_rad)
# print(temp_label)
temp_pred_rad_ = temp_pred_rad
temp_pred_rad[temp_pred_rad>=0.65] = 1
temp_pred_rad[temp_pred_rad<0.65] = 0
cm = confusion_matrix(temp_pred_rad, temp_label)
# print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Radiomics alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)

temp_pred_ssl_=temp_pred_ssl
temp_pred_ssl[temp_pred_ssl>=0.5] = 1
temp_pred_ssl[temp_pred_ssl<0.5] = 0
cm = confusion_matrix(temp_pred_ssl, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)

temp_pred_ssl_KMS_=temp_pred_ssl_KMS
temp_pred_ssl_KMS[temp_pred_ssl_KMS>=0.5] = 1
temp_pred_ssl_KMS[temp_pred_ssl_KMS<0.5] = 0
cm = confusion_matrix(temp_pred_ssl_KMS, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL-KMS alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)

temp_pred_ssl_RW_=temp_pred_ssl_RW
temp_pred_ssl_RW[temp_pred_ssl_RW>=0.5] = 1
temp_pred_ssl_RW[temp_pred_ssl_RW<0.5] = 0
cm = confusion_matrix(temp_pred_ssl_RW, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL-RW alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)

temp_pred_combined_= temp_pred_combined
temp_pred_combined[temp_pred_combined>=0.5] = 1
temp_pred_combined[temp_pred_combined<0.5] = 0
cm = confusion_matrix(temp_pred_combined, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)


temp_pred_combined_2_= temp_pred_combined_2
temp_pred_combined_2[temp_pred_combined_2>=0.6] = 1
temp_pred_combined_2[temp_pred_combined_2<0.6] = 0
cm = confusion_matrix(temp_pred_combined_2, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined with KMS:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)


temp_pred_combined_3_= temp_pred_combined_3
temp_pred_combined_3[temp_pred_combined_3>=0.6] = 1
temp_pred_combined_3[temp_pred_combined_3<0.6] = 0
cm = confusion_matrix(temp_pred_combined_3, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined with RW:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)

# TODO: Maybe include the plotting etc later
ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'only_l2_brats')
train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_ssl_features.npy'))
test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
# TODO: Should we load train_y_ from our splits???
# Normalize the features
for ii in range(np.shape(train_X_vit)[1]):
    train_X_vit[:, ii] = min_max_normalize(train_X_vit[:, ii], 1)
    test_X_vit[:, ii] = min_max_normalize(test_X_vit[:, ii], 1)

temp_pred_vit = classification(train_features=train_X_vit, train_label=train_y_, test_features=test_X_vit)
temp_pred_vit = temp_pred_vit[:, 1]



temp_pred_vit[temp_pred_vit>=0.5] = 1
temp_pred_vit[temp_pred_vit<0.5] = 0
cm = confusion_matrix(temp_pred_vit, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)


# Radiomics Combined with SSL
# TODO: Maybe include the plotting etc later
ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'only_l2_brats')
train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_ssl_features.npy'))
test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))

train_X_combined_3_ = np.concatenate((train_X_rad_, train_X_vit), axis=-1)
test_X_combined_3_ = np.concatenate((test_X_rad_, test_X_vit), axis=-1)


# TODO: Should we load train_y_ from our splits???
# Normalize the features
for ii in range(np.shape(train_X_vit)[1]):
    train_X_combined_3_[:, ii] = min_max_normalize(train_X_combined_3_[:, ii], 1)
    test_X_combined_3_[:, ii] = min_max_normalize(test_X_combined_3_[:, ii], 1)

temp_pred_vit = classification(train_features=train_X_combined_3_, train_label=train_y_, test_features=test_X_combined_3_)
temp_pred_vit = temp_pred_vit[:, 1]



temp_pred_vit[temp_pred_vit>=0.5] = 1
temp_pred_vit[temp_pred_vit<0.5] = 0
cm = confusion_matrix(temp_pred_vit, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL Combined with Radiomics:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)


# SSL Perceptual
print("SSL Perceptual")
# TODO: Maybe include the plotting etc later
ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps')
train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_ssl_features.npy'))
test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))
# TODO: Should we load train_y_ from our splits???
# Normalize the features
for ii in range(np.shape(train_X_vit)[1]):
    train_X_vit[:, ii] = min_max_normalize(train_X_vit[:, ii], 1)
    test_X_vit[:, ii] = min_max_normalize(test_X_vit[:, ii], 1)

temp_pred_vit = classification(train_features=train_X_vit, train_label=train_y_, test_features=test_X_vit)
temp_pred_vit = temp_pred_vit[:, 1]



temp_pred_vit[temp_pred_vit>=0.5] = 1
temp_pred_vit[temp_pred_vit<0.5] = 0
cm = confusion_matrix(temp_pred_vit, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL Perceptual:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)


# Radiomics Combined with SSL
# TODO: Maybe include the plotting etc later
ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, 'brats', 'ssl_features_dir', 'all_comps')
train_X_vit = np.load(os.path.join(ssl_feature_dir, 'train_ssl_features.npy'))
test_X_vit = np.load(os.path.join(ssl_feature_dir, 'test_ssl_features.npy'))

train_X_combined_3_ = np.concatenate((train_X_rad_, train_X_vit), axis=-1)
test_X_combined_3_ = np.concatenate((test_X_rad_, test_X_vit), axis=-1)


# TODO: Should we load train_y_ from our splits???
# Normalize the features
for ii in range(np.shape(train_X_vit)[1]):
    train_X_combined_3_[:, ii] = min_max_normalize(train_X_combined_3_[:, ii], 1)
    test_X_combined_3_[:, ii] = min_max_normalize(test_X_combined_3_[:, ii], 1)

temp_pred_vit = classification(train_features=train_X_combined_3_, train_label=train_y_, test_features=test_X_combined_3_)
temp_pred_vit = temp_pred_vit[:, 1]



temp_pred_vit[temp_pred_vit>=0.5] = 1
temp_pred_vit[temp_pred_vit<0.5] = 0
cm = confusion_matrix(temp_pred_vit, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL Perceptual Combined with Radiomics:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)