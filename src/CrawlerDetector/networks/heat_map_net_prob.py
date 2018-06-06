import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class HeatMapNetProb(NetworkBase):
    def __init__(self, conv_dim=16, repeat_num=1):
        super(HeatMapNetProb, self).__init__()
        self._name = 'HeatMapNetProb'

        self._heatmap_reg = self._create_heatmap_reg(conv_dim, repeat_num)
        self._prob_reg = self._create_prob_reg(conv_dim)

    def _create_heatmap_reg(self, conv_dim, repeat_num):
        layers = []
        # print conv_dim
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        down_dims_in = [conv_dim, conv_dim, conv_dim * 2, conv_dim * 2, conv_dim * 4]
        down_dims_out = [conv_dim, conv_dim * 2, conv_dim * 2, conv_dim * 4, conv_dim * 4]
        for i in range(len(down_dims_in)):
            # print 'd', down_dims_out[i]
            layers.append(nn.Conv2d(down_dims_in[i], down_dims_out[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(down_dims_out[i], affine=True))
            layers.append(nn.ReLU(inplace=True))

        # Bottleneck
        for i in range(repeat_num):
            # print 'b', conv_dim * 4
            layers.append(ResidualBlock(dim_in=conv_dim * 4, dim_out=conv_dim * 4))

        # Up-Sampling
        up_dims_in = [conv_dim * 4, conv_dim * 4, conv_dim * 2, conv_dim * 2, conv_dim]
        up_dims_out = [conv_dim * 4, conv_dim * 2, conv_dim * 2, conv_dim, conv_dim]
        for i in range(len(up_dims_in)):
            # print 'u', up_dims_out[i]
            layers.append(nn.ConvTranspose2d(up_dims_in[i], up_dims_out[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(up_dims_out[i], affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        return nn.Sequential(*layers)

    def _create_prob_reg(self, conv_dim):
        layers = []
        # print conv_dim
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        down_dims_in = [conv_dim, conv_dim, conv_dim * 2, conv_dim * 2, conv_dim * 4]
        down_dims_out = [conv_dim, conv_dim * 2, conv_dim * 2, conv_dim * 4, conv_dim * 4]
        for i in range(len(down_dims_in)):
            # print 'd', down_dims_out[i]
            layers.append(nn.Conv2d(down_dims_in[i], down_dims_out[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(down_dims_out[i], affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim * 4, conv_dim * 4, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim * 4, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim * 4, conv_dim * 4, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.Conv2d(conv_dim * 4, 1, kernel_size=1, stride=1, padding=0, bias=False))
        # layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self._heatmap_reg(x), self._prob_reg(x).view(x.size(0), -1)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=5, stride=1, padding=2, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)