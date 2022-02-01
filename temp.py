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
    x = np.arange(0, 10, 0.1)
    temp = [0, 0.4, 0.8, 1.2, 1.6, 2.0]
    for t in temp:
        exp_x = np.exp(x / t)
        vals = exp_x / exp_x.sum()
        plt.plot(vals, label=f'temperature: {t}')
    plt.legend(loc='best')
    plt.show()


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


if __name__ == '__main__':
    # plot_sin_cos()
    # plot_f_x()
    # vals = get_2d_sincos_pos_embed(embed_dim=8, grid_size=4, cls_token=True)
    # print(vals.shape)
    check_temp_param_effect()
