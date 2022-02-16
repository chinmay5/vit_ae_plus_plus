import os

import numpy as np
from sklearn import metrics
from sklearn import svm

from environment_setup import PROJECT_ROOT_DIR

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


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

# This is for the transfer approach testing

base_dir = '/mnt/cat/chinmay/brats_processed'
radiomics_path = os.path.join(base_dir, 'data', 'radiomics_features', 'features_flair.npy')
f1_path = os.path.join(base_dir, 'data', 'SS_features_45_orig_imbalanced.npy')
f2_path = os.path.join(base_dir, 'data', 'SS_features_k_means_k3_b6250.npy')
f3_path = os.path.join(base_dir, 'data', 'SS_features_k_means_k3_b61450_re-weighting.npy')
labels_path = os.path.join(base_dir, 'label_all.npy')

radiomics = np.asarray(np.load(radiomics_path))
epochs = 45
# Let us load the transfer features to check
contrast_ssl_feature_dir = os.path.join(PROJECT_ROOT_DIR, '', 'ssl_features_dir', 'transfer')
f4_path = os.path.join(contrast_ssl_feature_dir, 'features.npy')

f = np.load(f1_path)
f_2 = np.load(f2_path)
f_3 = np.load(f3_path)
f_4 = np.load(f4_path)


#f_2 = np.load('data/SS_features_k'+str(3)+'b_'+str(6)+'_epoch_'+str(2950)+'_256.npy')
#f_3 = np.load('data/SS_features_additional_'+'b_'+str(8)+'_epoch_'+str(1800)+'_256.npy')
#f = np.load('SS_features_k_means_b_8_'+str(epochs)+'_256.npy')

for ii in range(np.shape(radiomics)[1]):
#    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
    radiomics[:, ii] = min_max_normalize(radiomics[:, ii], 1)

for ii in range(np.shape(f)[1]):
    # f[:, ii] = z_score_normalize(f[:, ii])
    # f_2[:, ii] = z_score_normalize(f_2[:, ii])
    f[:, ii] = min_max_normalize(f[:, ii], 1)
    f_2[:, ii] = min_max_normalize(f_2[:, ii], 1)
    f_3[:, ii] = min_max_normalize(f_3[:, ii], 1)
    f_4[:, ii] = min_max_normalize(f_4[:, ii], 1)

# print(np.shape(radiomics))
labels = np.load(labels_path)
skf = StratifiedKFold(n_splits=5)
skf_50 = StratifiedKFold(n_splits=2)
#print(np.shape(skf))

def classification(train_features, train_label, test_features):
    clf = svm.SVC(gamma='auto', C=1, class_weight='balanced', probability=True, kernel='linear',)
    clf.fit(train_features, train_label)
    pred = clf.predict_proba(test_features)
    return pred

count = 0
for train, val in skf.split(radiomics, labels):

    train_X_rad_ = radiomics[train]
    train_X_ssl_ = f[train]
    train_X_ssl_KMS_ = f_2[train]
    train_X_ssl_RW_ = f_3[train]
    train_X_ssl_Contrast_ = f_4[train]

    train_X_combined_ = np.concatenate((train_X_rad_, train_X_ssl_), axis=-1)
    train_X_combined_2_ = np.concatenate((train_X_rad_, train_X_ssl_KMS_), axis=-1)
    train_X_combined_3_ = np.concatenate((train_X_rad_, train_X_ssl_RW_), axis=-1)
    train_X_combined_4_ = np.concatenate((train_X_rad_, train_X_ssl_Contrast_), axis=-1)

    train_y_ = labels[train]


    for train_50, val_50 in skf_50.split(train_X_rad_, train_y_):
        # print(np.shape(train_X_rad_))
        # print(np.shape(train_50))

        train_X_rad = train_X_rad_[train_50]
        train_X_ssl = train_X_ssl_[train_50]
        train_X_ssl_KMS = train_X_ssl_KMS_[train_50]
        train_X_ssl_RW = train_X_ssl_RW_[train_50]
        train_X_ssl_Contrast = train_X_ssl_Contrast_[train_50]

        train_X_combined = train_X_combined_[train_50]
        train_X_combined_2 = train_X_combined_2_[train_50]
        train_X_combined_3 = train_X_combined_3_[train_50]
        train_X_combined_4 = train_X_combined_4_[train_50]
        train_y = train_y_[train_50]


    val_X_rad = radiomics[val]
    val_X_ssl = f[val]
    val_X_ssl_KMS = f_2[val]
    val_X_ssl_RW = f_3[val]
    val_X_ssl_Contrast = f_4[val]
    #val_X_ssl_KMS_add = f_3[val]

    val_X_combined = np.concatenate((val_X_rad, val_X_ssl), axis=-1)
    val_X_combined_2 = np.concatenate((val_X_rad, val_X_ssl_KMS), axis=-1)
    val_X_combined_3 = np.concatenate((val_X_rad, val_X_ssl_RW), axis=-1)
    val_X_combined_4 = np.concatenate((val_X_rad, val_X_ssl_Contrast), axis=-1)
    #val_X_combined_3 = np.concatenate((val_X_rad, val_X_ssl_KMS_add), axis=-1)

    val_y = labels[val]

    if count == 0:
        temp_label = val_y
        temp_pred_rad = classification(train_X_rad, train_y, val_X_rad)
        temp_pred_ssl = classification(train_X_ssl, train_y, val_X_ssl)
        temp_pred_ssl_KMS = classification(train_X_ssl_KMS, train_y, val_X_ssl_KMS)
        temp_pred_ssl_RW = classification(train_X_ssl_RW, train_y, val_X_ssl_RW)
        temp_pred_ssl_Contrast = classification(train_X_ssl_Contrast, train_y, val_X_ssl_Contrast)

        temp_pred_combined = classification(train_X_combined, train_y, val_X_combined)
        temp_pred_combined_2 = classification(train_X_combined_2, train_y, val_X_combined_2)
        temp_pred_combined_3 = classification(train_X_combined_3, train_y, val_X_combined_3)
        temp_pred_combined_4 = classification(train_X_combined_4, train_y, val_X_combined_4)
     #   temp_pred_combined_3 = classification(train_X_combined_3, train_y, val_X_combined_3)

    else:
        test_pred_rad = classification(train_X_rad, train_y, val_X_rad)
        test_pred_ssl = classification(train_X_ssl, train_y, val_X_ssl)
        test_pred_ssl_KMS = classification(train_X_ssl_KMS, train_y, val_X_ssl_KMS)
        test_pred_ssl_RW = classification(train_X_ssl_RW, train_y, val_X_ssl_RW)
        test_pred_ssl_Contrast = classification(train_X_ssl_Contrast, train_y, val_X_ssl_Contrast)
        test_pred_combined = classification(train_X_combined, train_y, val_X_combined)
        test_pred_combined_2 = classification(train_X_combined_2, train_y, val_X_combined_2)
        test_pred_combined_3 = classification(train_X_combined_3, train_y, val_X_combined_3)
        test_pred_combined_4 = classification(train_X_combined_4, train_y, val_X_combined_4)
      #  test_pred_combined_3 = classification(train_X_combined_3, train_y, val_X_combined_3)

        temp_label = np.concatenate((temp_label, val_y), axis=0)
        temp_pred_rad = np.concatenate((temp_pred_rad, test_pred_rad), axis=0)
        temp_pred_ssl = np.concatenate((temp_pred_ssl, test_pred_ssl), axis=0)
        temp_pred_ssl_KMS = np.concatenate((temp_pred_ssl_KMS, test_pred_ssl_KMS), axis=0)
        temp_pred_ssl_RW = np.concatenate((temp_pred_ssl_RW, test_pred_ssl_RW), axis=0)
        temp_pred_ssl_Contrast = np.concatenate((temp_pred_ssl_Contrast, test_pred_ssl_Contrast), axis=0)
        temp_pred_combined = np.concatenate((temp_pred_combined, test_pred_combined), axis=0)
        temp_pred_combined_2 = np.concatenate((temp_pred_combined_2, test_pred_combined_2), axis=0)
        temp_pred_combined_3 = np.concatenate((temp_pred_combined_3, test_pred_combined_3), axis=0)
        temp_pred_combined_4 = np.concatenate((temp_pred_combined_4, test_pred_combined_4), axis=0)
       # temp_pred_combined_3 = np.concatenate((temp_pred_combined_3, test_pred_combined_3), axis=0)

    count += 1

temp_label = np.asarray(temp_label)
temp_pred_rad = temp_pred_rad[:, 1]
temp_pred_ssl = temp_pred_ssl[:, 1]
temp_pred_ssl_KMS = temp_pred_ssl_KMS[:, 1]
temp_pred_ssl_RW = temp_pred_ssl_RW[:, 1]
temp_pred_ssl_Contrast = temp_pred_ssl_Contrast[:, 1]
temp_pred_combined = temp_pred_combined[:, 1]
temp_pred_combined_2 = temp_pred_combined_2[:, 1]
temp_pred_combined_3 = temp_pred_combined_3[:, 1]
temp_pred_combined_4 = temp_pred_combined_4[:, 1]


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

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_ssl_Contrast)
auc = metrics.roc_auc_score(temp_label, temp_pred_ssl_Contrast)
plt.plot(fpr,tpr,label="SSL+Contrast radiomics, AUC = "+str(auc)[0:5])
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

fpr, tpr, _ = metrics.roc_curve(temp_label,  temp_pred_combined_4)
auc = metrics.roc_auc_score(temp_label, temp_pred_combined_4)
plt.plot(fpr,tpr,label="trad. + SSL-Contrast radiomics, AUC = "+str(auc)[0:5])
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
roc_auc_score_rad = roc_auc_score(temp_label, temp_pred_rad_)

temp_pred_rad[temp_pred_rad>=0.65] = 1
temp_pred_rad[temp_pred_rad<0.65] = 0
cm = confusion_matrix(temp_pred_rad, temp_label)
# print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Radiomics alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_rad)


temp_pred_ssl_=temp_pred_ssl
roc_auc_score_ssl = roc_auc_score(temp_label, temp_pred_ssl_)
temp_pred_ssl[temp_pred_ssl>=0.5] = 1
temp_pred_ssl[temp_pred_ssl<0.5] = 0
cm = confusion_matrix(temp_pred_ssl, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_ssl)

temp_pred_ssl_KMS_=temp_pred_ssl_KMS
roc_auc_score_KMS = roc_auc_score(temp_label, temp_pred_ssl_KMS_)
temp_pred_ssl_KMS[temp_pred_ssl_KMS>=0.5] = 1
temp_pred_ssl_KMS[temp_pred_ssl_KMS<0.5] = 0
cm = confusion_matrix(temp_pred_ssl_KMS, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL-KMS alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_KMS)

temp_pred_ssl_RW_=temp_pred_ssl_RW
roc_auc_score_RW = roc_auc_score(temp_label, temp_pred_ssl_RW_)
temp_pred_ssl_RW[temp_pred_ssl_RW>=0.5] = 1
temp_pred_ssl_RW[temp_pred_ssl_RW<0.5] = 0
cm = confusion_matrix(temp_pred_ssl_RW, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL-RW alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_RW)


temp_pred_ssl_Contrast_=temp_pred_ssl_Contrast
roc_auc_score_Contrast = roc_auc_score(temp_label, temp_pred_ssl_Contrast_)
temp_pred_ssl_Contrast_[temp_pred_ssl_Contrast_>=0.5] = 1
temp_pred_ssl_Contrast_[temp_pred_ssl_Contrast_<0.5] = 0
cm = confusion_matrix(temp_pred_ssl_Contrast_, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('SSL-Contrast alone:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_Contrast)



temp_pred_combined_= temp_pred_combined
roc_auc_score_combined = roc_auc_score(temp_label, temp_pred_combined_)
temp_pred_combined[temp_pred_combined>=0.5] = 1
temp_pred_combined[temp_pred_combined<0.5] = 0
cm = confusion_matrix(temp_pred_combined, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score:", roc_auc_score_combined)

temp_pred_combined_2_= temp_pred_combined_2
roc_auc_score_combined_2 = roc_auc_score(temp_label, temp_pred_combined_2_)
temp_pred_combined_2[temp_pred_combined_2>=0.6] = 1
temp_pred_combined_2[temp_pred_combined_2<0.6] = 0
cm = confusion_matrix(temp_pred_combined_2, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined with KMS:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score", roc_auc_score_combined_2)

temp_pred_combined_3_= temp_pred_combined_3
roc_auc_score_combined_3 = roc_auc_score(temp_label, temp_pred_combined_3)
temp_pred_combined_3[temp_pred_combined_3>=0.6] = 1
temp_pred_combined_3[temp_pred_combined_3<0.6] = 0
cm = confusion_matrix(temp_pred_combined_3, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined with RW:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score", roc_auc_score_combined_3)



temp_pred_combined_4_= temp_pred_combined_4
roc_auc_score_combined_4 = roc_auc_score(temp_label, temp_pred_combined_4)
temp_pred_combined_4[temp_pred_combined_4>=0.6] = 1
temp_pred_combined_4[temp_pred_combined_4<0.6] = 0
cm = confusion_matrix(temp_pred_combined_4, temp_label)
print(cm)
specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
print('Combined with Contrast:')
print('specificity:', specificity)
sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
print('sensitivity:', sensitivity)
print("roc_auc_score", roc_auc_score_combined_4)


# temp_pred_combined_3_= temp_pred_combined_3
# temp_pred_combined_3[temp_pred_combined_3>=0.6] = 1
# temp_pred_combined_3[temp_pred_combined_3<0.6] = 0
# cm = confusion_matrix(temp_pred_combined_3, temp_label)
# print(cm)
# specificity= cm[0, 0]/(cm[0, 0]+cm[1, 0])
# print('Combined vs:')
# print('specificity:', specificity)
# sensitivity =  cm[1, 1]/(cm[1, 1]+cm[0, 1])
# print('sensitivity:', sensitivity)
#

#
# from scipy.stats import wilcoxon
# # seed the random number generator
# # compare samples
# temp_pred_combined_2_ = np.concatenate((temp_pred_combined_2_[labels==0], -temp_pred_combined_2_[labels==1]), axis=0)
# temp_pred_ssl_ = np.concatenate((temp_pred_ssl_[labels==0], -temp_pred_ssl_[labels==1]), axis=0)
# temp_pred_rad_ = np.concatenate((temp_pred_rad_[labels==0], -temp_pred_rad_[labels==1]), axis=0)
#
# stat, p = wilcoxon(temp_pred_rad_, temp_pred_combined_2_)
# print('p value: combined vs. trad.', p)
# stat, p = wilcoxon(temp_pred_ssl_, temp_pred_combined_2_)
# print('p value: combined vs. ssl.', p)


# stat, p = wilcoxon(temp_pred_rad, temp_pred_ssl)
# print('p value: combined vs. trad.', p)
# 9

# print(np.shape(f))
# print(np.shape(labels))
