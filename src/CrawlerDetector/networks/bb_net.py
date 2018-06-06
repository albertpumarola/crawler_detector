import torch.nn as nn
from .networks import NetworkBase
import torch

class BBNet(NetworkBase):

    def __init__(self, freeze=False):
        super(BBNet, self).__init__()

        self._bb_estim = self._make_layers()
        self._init_weights()
        self._set_requires_grads(self._bb_estim, requires_grads=(not freeze))

    def _make_layers(self, n_upsampling=1):
        layers = []
        cfg = [512, 512, 256, 128]
        for i in xrange(len(cfg)-1):
            layers.append(nn.ConvTranspose2d(cfg[i], cfg[i+1],
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=False))
            layers.append(nn.BatchNorm2d(cfg[i+1]))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(cfg[-1], 2, kernel_size=1, stride=1))
        layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, x):
        # Bx[512, 512, 512, 256, 128, 2]x[7, 14, 28, 56, 56]x[7, 14, 28, 56, 56]
        return self._bb_estim(x)