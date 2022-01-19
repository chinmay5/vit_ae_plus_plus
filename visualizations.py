import os

import numpy as np
import torch
from dotmap import DotMap

from dataset.pretrain_tumor_data import build_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_factory import get_models

import os
import nibabel as nib

save_path = os.path.join(PROJECT_ROOT_DIR, 'temp')
os.makedirs(save_path, exist_ok=True)


def prepare_model():
    # build model
    args = DotMap()
    args.volume_size = 96
    args.in_channels = 1
    args.nb_classes = 2
    args.patch_size = 8
    args.model = 'mae_vit_base_patch16'
    args.finetune = os.path.join(PROJECT_ROOT_DIR, 'output_dir', 'checkpoints', 'checkpoint-380.pth')

    model = get_models(model_name='autoenc', args=args)
    # load model
    checkpoint = torch.load(args.finetune, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % args.finetune)
    checkpoint_model = checkpoint['model']
    msg = model.load_state_dict(checkpoint_model, strict=True)
    print(msg)
    return model


def viz_one_patch(model):
    dataset_test = build_dataset(mode='test')
    x, _ = dataset_test[0]
    x.unsqueeze_(0)  # Adding the batch dimension
    # make it a batch-like

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    # y = torch.einsum('nclhw->nlhwc', y).detach().cpu()
    y = y.detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 3 * 1)  # (N, L*H*W, p*p*1) 1=num_channel
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nclhw->nlhwc', mask).detach().cpu()
    mask = mask.detach().cpu()

    # x = torch.einsum('nclhw->nlhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    save_nifty_img(image=im_masked, file_name='masked.nii.gz')

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    save_nifty_img(image=im_paste, file_name='recons_und_visible.nii.gz')

    # Only the reconstruction
    save_nifty_img(image=y, file_name='reconstruct.nii.gz')


def save_nifty_img(image, file_name):
    # We need to scale the values so that they lie in [0, 1]
    img = (image.numpy()[0] + 1) / 2 # original [-1, 1], now [0, 1]
    img_scaled = 255 * img
    img_scaled = img_scaled[0]  # (1, 1, 96, 96, 96) -> We take the first element form the batch and the values of its only channel.
    img_pasted = nib.Nifti1Image(img_scaled.astype(np.int8), np.eye(4))
    print(f"Saving file {file_name}")
    nib.save(img_pasted, os.path.join(save_path, file_name))


def save_samples():
    model = prepare_model()
    viz_one_patch(model=model)


if __name__ == '__main__':
    save_samples()
