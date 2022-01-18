import torch
from sklearn.metrics import confusion_matrix, roc_auc_score


def roc_auc(predictions, target):
    predictions = torch.sigmoid(predictions)
    predictions = predictions[:, 1]
    predictions, target = predictions.cpu().numpy(), target.cpu().numpy()
    # Arguments take 'GT' before taking 'predictions'
    return roc_auc_score(target, predictions)


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
    x = torch.randn((4, 2))
    y = torch.tensor([0, 1, 1, 0])
    print(roc_auc(x, y))