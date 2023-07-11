import os
import numpy as np
from matplotlib import pyplot as plt
import torchio as tio

BASE_PATH = '/mnt/cat/chinmay/brats_processed/data/image'


def read_data_volume():
    file_path = os.path.join(BASE_PATH, 'flair_all.npy')
    data = np.load(file_path)
    return data

def plot_one_slice():
    data =read_data_volume()
    print(f"Original shape {data.shape}")
    data = data.transpose([0, 4, 1, 2, 3])[0]  # Select one sample
    transforms = [
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ]
    transformations = tio.Compose(transforms)
    data = transformations(data)
    one_slice = data[0, 64]  # would be the shape 96, 96
    plt.imshow(one_slice)
    plt.show()

if __name__ == '__main__':
    plot_one_slice()
