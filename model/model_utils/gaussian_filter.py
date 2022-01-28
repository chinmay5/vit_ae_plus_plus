import torch
from torch.nn import functional as F


def make_gaussian_kernel(sigma):
    ks = int(sigma * 5)
    if ks % 2 == 0:
        ks += 1
    ts = torch.linspace(-ks // 2, ks // 2 + 1, ks)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def perform_3d_gaussian_blur(input_vol, blur_sigma=2):
    _, in_ch, _, _, _ = input_vol.shape
    assert in_ch == 1, "Works only for Grayscale images"
    k = make_gaussian_kernel(blur_sigma)
    k3d = torch.einsum('i,j,k->ijk', k, k, k).to(input_vol.device)
    k3d = k3d / k3d.sum()
    vol_3d = F.conv3d(input_vol, k3d.reshape(in_ch, in_ch, *k3d.shape), stride=1, padding=len(k) // 2)
    return vol_3d


if __name__ == '__main__':
    # filter = GaussianSmoothing(channels=1, kernel_size=3, sigma=1, dim=1)
    # x = torch.rand(1, 10, 10)
    # output = filter(x)
    input_vol = torch.randn(4, 1, 32, 32, 32)
    print(perform_3d_gaussian_blur(input_vol=input_vol).shape)
