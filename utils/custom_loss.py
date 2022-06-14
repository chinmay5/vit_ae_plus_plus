import torch
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from torch import nn


class SoftCrossEntropyWithWeightsLoss(nn.Module):
    def __init__(self, weights):
        super(SoftCrossEntropyWithWeightsLoss, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad=False)

    def forward(self, y_hat, y):
        weighted_logits = F.log_softmax(y_hat, dim=-1) * self.weights
        # The choice of dim is important.
        # Remember, we need to work on N x D matrix and thus, w1 x11 + w2 x12 +...wN x1N / w1 + ...wN => dim=0
        # for the summation
        weighted_sum = torch.sum(-y * weighted_logits, dim=0) / self.weights.sum()
        return weighted_sum.mean()

    def __repr__(self):
        return f"weights are on {self.weights.device}\n"


def check_implementation():
    loss_fn_custom = SoftCrossEntropyWithWeightsLoss(weights=torch.as_tensor([1, 1]))
    loss_fn_orig = SoftTargetCrossEntropy()
    y_hat = torch.randn(4, 2)
    y = torch.rand(4, 2)
    assert (torch.allclose(loss_fn_custom(y_hat,y), loss_fn_orig(y_hat, y))), "Something wrong with implementation"
    print("Works!!!")


if __name__ == '__main__':
    loss_fn = SoftCrossEntropyWithWeightsLoss(weights=torch.as_tensor([100, 300]))
    torch.random.manual_seed(42)
    y_hat = torch.randn(4, 2)
    y = torch.rand(4, 2)
    print(loss_fn(y_hat, y))
    check_implementation()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = SoftCrossEntropyWithWeightsLoss(weights=torch.as_tensor([100, 300])).cuda()
    print(loss_fn)
