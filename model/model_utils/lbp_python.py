import numpy as np
from scipy.spatial.distance import euclidean
from skimage.feature import local_binary_pattern
from skimage.feature._texture import _local_binary_pattern


def lbp_histogram(volume):
    # patterns = local_binary_pattern(volume, 8, 1)
    image = np.ascontiguousarray(volume, dtype=np.double)
    patterns = _local_binary_pattern(image, P=8, R=1, method='D')
    hist, _ = np.histogram(patterns, bins=np.arange(2**8 + 1), density=True)
    return hist

x = np.random.randn(96, 96, 96)
lbp_histogram(x)