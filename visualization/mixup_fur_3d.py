import os

import torch
from matplotlib import pyplot as plt
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from dataset.brain_tumor.pretrain_tumor_data import build_dataset
from timm.data.mixup import Mixup

from environment_setup import PROJECT_ROOT_DIR
from visualization.sanity_checks import plot_img_util

log_dir = os.path.join(PROJECT_ROOT_DIR, 'temp')
log_writer = SummaryWriter(log_dir)


def sample_from_beta(alpha):
    beta_dist = Beta(alpha, alpha)
    samples = []
    for i in range(10000):
        samples.append(beta_dist.sample().item())
    return samples

def check_mixup_fur_3d():
    dataset_test = build_dataset(mode='test', use_z_score=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    x, tar = next(iter(data_loader_test))
    x, tar = x.cuda(), tar.cuda()
    mixup_fn = Mixup(
        mixup_alpha=0.8, num_classes=2)
    for _ in range(10):
        samples, targets = mixup_fn(x, tar)
        print(targets)
    # Visualize the resulting images. If this works, we can use it directly as a regularization technique.
    # samples B, N, L, H, W
    cutmix_img = plot_img_util(samples[0][0].squeeze_().unsqueeze(1))
    print(targets)
    log_writer.add_images(tag=f'cutmix_img-{targets[0]}', img_tensor=cutmix_img)


if __name__ == '__main__':
    check_mixup_fur_3d()
    # for alpha in torch.arange(0.1, 0.5, step=0.1):
    #     samples = sample_from_beta(alpha=alpha)
    #     print(f"Mean for {alpha} is {torch.as_tensor(samples).mean()}")
