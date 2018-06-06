import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch
import torch.utils.model_zoo as model_zoo

class uvProbNet(NetworkBase):
    def __init__(self, num_nc=32, do_add_batchnorm=False):
        super(uvProbNet, self).__init__()
        self._name = 'uvProbNet'

        features_cfg = [num_nc, num_nc, 'M', 2*num_nc, 2*num_nc, 'M', 4*num_nc, 4*num_nc,  'M', 6*num_nc, 6*num_nc,  'M', 6*num_nc, 6*num_nc, 'M']

        self._features = self._make_layers(features_cfg, 3, do_add_batchnorm)
        # self._pose_conv = self._make_layers(pose_cfg, 4*num_nc, batch_norm=False)
        # self._prob_conv = self._make_layers(prob_cfg, 4*num_nc, batch_norm=False)

        self._pose_reg = self._make_reg(2, 6*num_nc)
        self._prob_reg = self._make_reg(1, 6*num_nc)

    def _make_layers(self, cfg, in_channels, batch_norm):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _make_reg(self, num_classes, nc):
        return nn.Sequential(
            nn.Linear(nc * 7 * 7, nc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nc, nc),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nc, num_classes),
        )

    def forward(self, x):
        features = self._features(x)
        pose = self._pose_reg(features.view(x.size(0), -1))
        prob = self._prob_reg(features.view(x.size(0), -1))
        return pose, prob