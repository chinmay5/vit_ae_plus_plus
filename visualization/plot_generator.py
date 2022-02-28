import matplotlib.pyplot as plt
import numpy as np


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    print(f"Pos embed shape {pos_embed.shape}")
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    print(f"First grid shape {emb_h.shape}")
    print(f"Second grid shape {emb_w.shape}")
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def check_temp_param_effect():
    temp = [0.01, 0.1, 1, 10, 100]
    logits = np.asarray([-0.4, -0.2, 0.2, 0.4, 1, 2.0])
    for t in temp:
        expos = np.sum(np.exp(logits * t))
        prob = np.exp(logits * t) / expos
        print(f"For temperature {1 / t} the probability is {prob}")


def plot_f_x():
    # Assume an input sequence of 20 values used
    f_x = lambda x: 1 / (100 ** (2 * x / 64))
    x = np.arange(-5, 5, 0.01)
    for k in range(1, 10):
        y = [f_x(k * m) for m in x]
        plt.plot(y)
    plt.show()


def plot_sin_cos():
    values = np.arange(0, 6, 0.2)
    for x in range(0, 10):
        y = np.sin(x * values)
        plt.plot(y)
    plt.show()


def load_img(scan_name='MR_EGD-0774'):
    dir = '/mnt/cat/chinmay/glioma_Bene/pre_processed'
    import os
    flair = np.load(os.path.join(dir, scan_name, 'flair.npy'))
    t1ce = np.load(os.path.join(dir, scan_name, 't1ce.npy'))
    t1 = np.load(os.path.join(dir, scan_name, 't1.npy'))
    t2 = np.load(os.path.join(dir, scan_name, 't2.npy'))
    plt.imshow(flair[0])
    plt.show()
    plt.imshow(t1ce[0])
    plt.show()
    plt.imshow(t1[0])
    plt.show()
    plt.imshow(t2[0])
    plt.show()


def show_plot():
    import matplotlib.pyplot as plt
    from numpy.random import random
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 500

    import matplotlib.pyplot as plt

    lambda_ = [0.55, 0.65, 0.75, 0.85, 0.95]
    specificity = [0.61, 0.68, 0.761, 0.667, 0.668]
    sensitivity = [0.88, 0.84, 0.836, 0.851, 0.831]

    fig, ax = plt.subplots()

    ax.plot(lambda_, specificity, color='red', marker='*', alpha=0.45, linewidth=2.2, label='Specificity')
    ax.plot(lambda_, sensitivity, color='blue', marker='o', alpha=0.45, linewidth=2.2, label='Sensitivity')
    # ax.axis('equal')
    leg = ax.legend(fontsize=14)
    plt.title('', fontsize=14)
    plt.xlabel('', fontsize=14)
    plt.ylabel('score', fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()


if __name__ == '__main__':
    # plot_sin_cos()
    # plot_f_x()
    # vals = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4, cls_token=True)
    # print(vals.shape)
    show_plot()
