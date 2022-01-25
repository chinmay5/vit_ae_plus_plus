import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Need to convert the values in one-hot encoding
enc = OneHotEncoder()
possible_labels = np.array([0, 1]).reshape(-1, 1)
enc.fit(possible_labels)


def roc_auc(predictions, target):
    # Converting raw scores into probabilities
    specificity, sensitivity = find_vals(predictions, target)
    predictions = torch.softmax(predictions, dim=1)
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    target_one_hot = enc.transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target_one_hot, predictions), specificity, sensitivity


def find_vals(predictions, target):
    predictions = torch.max(predictions, dim=1)[1]  # We need the indices for the max
    print(predictions)
    print(target)
    cm = confusion_matrix(predictions.cpu().numpy(), target.cpu().numpy())
    print(cm)
    specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print('Combined:')
    print('specificity:', specificity)
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    print('sensitivity:', sensitivity)
    return specificity, sensitivity


if __name__ == '__main__':
    y = np.array([0, 1, 1, 0])
    # y = np.array([0, 1, 1, 0]).reshape(-1, 1)
    # enc.fit(possible_labels)
    # y = enc.transform(y).toarray()
    # print(y)
    x = torch.randn((4, 2))
    # y = torch.as_tensor(y).squeeze()
    y = torch.as_tensor(y)
    print(roc_auc(x, y))
    print(find_vals(x, y))
