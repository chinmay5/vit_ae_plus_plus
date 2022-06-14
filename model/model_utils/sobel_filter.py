import torch
from torch import nn


class SobelFilter3d(nn.Module):
    def __init__(self):
        super(SobelFilter3d, self).__init__()
        self.sobel_filter = self.setup_filter()

    def setup_filter(self):
        sobel_filter = nn.Conv3d(1, 3, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor(
                [
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                    [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                    [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
                ]))
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            ]))
        sobel_filter.weight.data[2, 0].copy_(
            torch.FloatTensor([
                [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            ])
        )
        sobel_filter.bias.data.zero_()
        for p in sobel_filter.parameters():
            p.requires_grad = False
        return sobel_filter

    def forward(self, x):
        bs, ch, l, h, w = x.shape
        combined_edge_map = 0
        for idx in range(ch):
            g_x = self.sobel_filter(x[:, idx:idx+1])[:, 0]
            g_y = self.sobel_filter(x[:, idx:idx+1])[:, 1]
            g_z = self.sobel_filter(x[:, idx:idx+1])[:, 2]
            combined_edge_map += torch.sqrt((g_x **2 + g_y ** 2 + g_z ** 2))
        return combined_edge_map


if __name__ == '__main__':
    filter = SobelFilter3d()
    x = torch.randn((4, 1, 32, 32, 32))
    output1 = filter(x)
    x = torch.randn((4, 3, 32, 32, 32))
    output2 = filter(x)
    loss = torch.nn.L1Loss()
    print(loss(output1, output2).sum())
