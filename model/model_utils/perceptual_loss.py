import os

import torch
from torchvision import models as tv

from collections import namedtuple

from environment_setup import PROJECT_ROOT_DIR


class vgg_perceptual_loss(torch.nn.Module):
    def __init__(self, requires_grad=False, use_imagenet=False):
        super(vgg_perceptual_loss, self).__init__()

        if use_imagenet:
            print("Using VGG-Imagenet pretrained model for perceptual loss")
            vgg_pretrained_features = tv.vgg16(pretrained=True).eval().features
        else:
            print("Using VGG SSL features for training")
            vgg_pretrained_model = tv.vgg16(pretrained=False)
            model_path = os.path.join(PROJECT_ROOT_DIR, 'model', 'ckp-399.pth')
            vgg_pretrained_model.load_state_dict(
                torch.load(model_path), strict=False)
            vgg_pretrained_features = vgg_pretrained_model.eval().features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        self.N_slices = 4
        self.mse_loss = torch.nn.MSELoss()

        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward_one_view(self, X):
        # X = bs, ch, z, y, x
        X = X.permute(0, 2, 1, 3, 4)  # bs, z, ch, y, x
        X = X.reshape(-1, *X.size()[2:])
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        return out

    def compute_loss_per_channel(self, X1, X2):
        view1_activations = self.forward_one_view(X1)
        view2_activations = self.forward_one_view(X2)
        loss = torch.mean(
            torch.as_tensor([self.mse_loss(view1_activations[i], view2_activations[i]) for i in range(self.N_slices)]))
        return loss

    def forward(self, X1, X2):
        bs, ch, _, _, _ = X1.shape
        loss = 0
        for idx in range(ch):
            loss += self.compute_loss_per_channel(X1[:, idx:idx+1], X2[:, idx:idx+1])
        return loss / ch

    # def forward(self, X1, X2):
    #     view1_activations = self.forward_one_view(X1)
    #     view2_activations = self.forward_one_view(X2)
    #     loss = torch.mean(
    #         torch.as_tensor([self.mse_loss(view1_activations[i], view2_activations[i]) for i in range(self.N_slices)]))
    #     return loss


if __name__ == '__main__':
    loss = vgg_perceptual_loss()
    volume1 = torch.randn(4, 4, 32, 32, 32)
    volume2 = torch.randn(4, 4, 32, 32, 32)
    print(loss(volume1, volume2))
