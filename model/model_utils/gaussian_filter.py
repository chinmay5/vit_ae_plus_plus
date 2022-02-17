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


def perform_3d_gaussian_blur(original_vol, blur_sigma=2):
    _, in_ch, _, _, _ = original_vol.shape
    return_vol = []
    for idx in range(in_ch):
        input_vol = original_vol[:, idx:idx+1]
        k = make_gaussian_kernel(blur_sigma)
        k3d = torch.einsum('i,j,k->ijk', k, k, k).to(input_vol.device)
        k3d = k3d / k3d.sum()
        vol_3d = F.conv3d(input_vol, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(k) // 2)
        return_vol.append(vol_3d)
    return torch.cat(return_vol, dim=1)


if __name__ == '__main__':
    # filter = GaussianSmoothing(channels=1, kernel_size=3, sigma=1, dim=1)
    # x = torch.rand(1, 10, 10)
    # output = filter(x)
    input_vol = torch.randn(4, 3, 32, 32, 32)
    print(perform_3d_gaussian_blur(original_vol=input_vol).shape)
