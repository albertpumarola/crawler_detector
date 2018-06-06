import torch
import torch.nn as nn
from .networks import NetworkBase


class BBNetValues(NetworkBase):

    def __init__(self, freeze=False):
        super(BBNetValues, self).__init__()

        self._down_scale = self._make_layers()
        self._estimator = self._make_estimator()

        self._init_weights()

        self._set_requires_grads(self._down_scale, requires_grads=(not freeze))
        self._set_requires_grads(self._estimator, requires_grads=(not freeze))

    def _make_layers(self):
        cfg = [512, 'M', 512, 'M']
        layers = []
        in_channels = 512

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def _make_estimator(self):
        return nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self._down_scale(x)
        x = x.view(x.size(0), -1)
        return self._estimator(x)