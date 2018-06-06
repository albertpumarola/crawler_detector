import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class HeatMapNetProb(NetworkBase):
    def __init__(self, conv_dim=16):
        super(HeatMapNetProb, self).__init__()
        self._name = 'HeatMapNetProb'

        self._create_heatmap_reg(conv_dim)
        self._prob_reg_feat = self._create_prob_reg_feat(conv_dim)
        self._prob_reg = self._create_prob_reg()

    def _create_heatmap_reg(self, conv_dim):
        self._seq1 = nn.Sequential(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.InstanceNorm2d(conv_dim, affine=True),
                                   nn.ReLU(inplace=True))

        self._down_seq1 = self._create_hm_dlayer(conv_dim, conv_dim)
        self._down_seq2 = self._create_hm_dlayer(conv_dim, 2*conv_dim)
        self._down_seq3 = self._create_hm_dlayer(2*conv_dim, 4*conv_dim)
        self._down_seq4 = self._create_hm_dlayer(4*conv_dim, 8*conv_dim)
        self._down_seq5 = self._create_hm_dlayer(8*conv_dim, 16*conv_dim)

        self._bottle_seq1 = ResidualBlock(dim_in=conv_dim * 16, dim_out=conv_dim * 16)

        self._up_seq1 = self._create_hm_ulayer(16*conv_dim, 8*conv_dim)
        self._up_seq2 = self._create_hm_ulayer(8 * conv_dim * 2, 4 * conv_dim)
        self._up_seq3 = self._create_hm_ulayer(4 * conv_dim * 2, 2 * conv_dim)
        self._up_seq4 = self._create_hm_ulayer(2 * conv_dim * 2, conv_dim)
        self._up_seq5 = self._create_hm_ulayer(conv_dim * 2, conv_dim)

        self._reg = nn.Conv2d(conv_dim, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def _create_hm_dlayer(self, input_dim, output_dim):
        layers = []
        layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _create_hm_ulayer(self, input_dim, output_dim):
        layers = []
        layers.append(nn.ConvTranspose2d(input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(output_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _create_prob_reg_feat(self, conv_dim):
        layers = []
        # print conv_dim
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        down_dims_in = [conv_dim, conv_dim, conv_dim * 2, conv_dim * 4, conv_dim * 8]
        down_dims_out = [conv_dim, conv_dim * 2, conv_dim * 4, conv_dim * 8, conv_dim * 8]
        for i in range(len(down_dims_in)):
            layers.append(nn.Conv2d(down_dims_in[i], down_dims_out[i], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(down_dims_out[i], affine=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=4, stride=1, padding=0, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim * 8, affine=True))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _create_prob_reg(self):
        layers = []
        layers.append(nn.Linear(128 * 4 * 4, 256))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU(True))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(256, 1))
        return nn.Sequential(*layers)

    def _forward_heatmap_reg(self, x):
        y1 = self._seq1(x)
        y2 = self._down_seq1(y1)
        y3 = self._down_seq2(y2)
        y4 = self._down_seq3(y3)
        y5 = self._down_seq4(y4)
        y6 = self._down_seq5(y5)

        y7 = self._bottle_seq1(y6)

        y8 = self._up_seq1(y7)
        y9 = self._up_seq2(torch.cat((y5, y8), 1))
        y10 = self._up_seq3(torch.cat((y4, y9), 1))
        y11 = self._up_seq4(torch.cat((y3, y10), 1))
        y12 = self._up_seq5(torch.cat((y2, y11), 1))

        y13 = self._reg(y12)
        return y13

    def forward(self, x):
        prob_feat = self._prob_reg_feat(x).view(x.size(0), -1)
        return self._forward_heatmap_reg(x), self._prob_reg(prob_feat)


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