import torch.nn as nn
from .networks import NetworkBase
import torch.utils.model_zoo as model_zoo

class VggFeatures(NetworkBase):

    def __init__(self, freeze=False):
        super(VggFeatures, self).__init__()

        self.features = self._make_layers()
        self.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'), strict=False)
        self._set_requires_grads(self.features, requires_grads=(not freeze))

    def _make_layers(self):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        # Bx[3, 64, 128, 256, 512, 512]x[224, 112, 56, 28, 14, 7]x[224, 112, 56, 28, 14, 7]
        return self.features(x)