import os

from torch.utils.tensorboard import SummaryWriter

from dataset.brain_tumor.pretrain_tumor_data import build_dataset
from environment_setup import PROJECT_ROOT_DIR
from model.model_utils.gaussian_filter import perform_3d_gaussian_blur
from model.model_utils.sobel_filter import SobelFilter3d
from visualization.sanity_checks import plot_img_util
from visualization.visualizations import prepare_model, save_nifty_img

log_dir = os.path.join(PROJECT_ROOT_DIR, 'temp')
train_writer = SummaryWriter(log_dir)


def check_reconstruct(input_volume, model, mask_ratio):
    # run MAE
    patched_input = model.patchify(input_volume)
    reconstructed_input = model.unpatchify(patched_input)
    return reconstructed_input


def get_data():
    dataset_test = build_dataset(mode='test', use_z_score=True)
    x, _ = dataset_test[0]
    x.unsqueeze_(0)  # Adding the batch dimension
    # make it a batch-like
    return x


def execute_reconstruction():
    x = get_data()
    model = prepare_model()
    reconstructed_input = check_reconstruct(input_volume=x, model=model, mask_ratio=0.75)
    save_nifty_img(image=x, file_name='original.nii.gz')
    save_nifty_img(image=reconstructed_input, file_name='patch_reconstruct.nii.gz')


def sobel_checks():
    x = get_data()
    filter = SobelFilter3d()
    output = filter(x)
    blurred_img = perform_3d_gaussian_blur(x)
    blurred_sobel = filter(blurred_img)
    output_images = plot_img_util(output[0].squeeze_().unsqueeze(1))
    input_images = plot_img_util(x[0].squeeze_().unsqueeze(1))
    blurred_sobel = plot_img_util(blurred_sobel[0].squeeze_().unsqueeze(1))
    blurred_images = plot_img_util(blurred_img[0].squeeze_().unsqueeze(1))
    train_writer.add_images(tag='sobel_out', img_tensor=output_images)
    train_writer.add_images(tag='input', img_tensor=input_images)
    train_writer.add_images(tag='blurred', img_tensor=blurred_images)
    train_writer.add_images(tag='blurred_sobel', img_tensor=blurred_sobel)


if __name__ == '__main__':
    # execute_reconstruction()
    sobel_checks()
