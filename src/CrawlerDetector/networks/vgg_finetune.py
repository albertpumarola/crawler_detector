import torch.nn as nn
from .networks import NetworkBase
import torchvision
import math

class VGG11(NetworkBase):
    def __init__(self, bb_nc=4, prob_nc=1):
        super(VGG11, self).__init__()
        self._bb_nc = bb_nc
        self._prob_nc = prob_nc
        self._model = torchvision.models.vgg11(pretrained=True)
        self._set_requires_grads(self._model, requires_grads=False)

        # num_ftrs = self._model.classifier[6].in_features
        # features = list(self._model.classifier.children())[:-1]
        # features.extend([nn.Linear(num_ftrs, bb_nc + prob_nc)])
        # self._model.classifier = nn.Sequential(*features)
        self._model = self._model.features
        self._bb_reg = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, bb_nc),
        )

        self._prob_reg = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, prob_nc),
            nn.Sigmoid()
        )

        # for name, param in self._model.named_parameters():
        #     print name, param.requires_grad

        # self._sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self._model(x)
        features = features.view(features.size(0), -1)
        bb = self._bb_reg(features)
        prob = self._prob_reg(features)
        return bb, prob
        # out = self._model(x)
        # bb = out[:, :self._bb_nc]
        # prob = out[:, self._bb_nc:]
        # return bb, prob

# class VGG11(NetworkBase):
#
#     def __init__(self, bb_nc=4, prob_nc=1):
#         super(VGG11, self).__init__()
#         self.features = self._make_conv_features()
#         self.classifier = self._make_lin_features()
#         self._bb_reg = self._make_bb_reg(bb_nc)
#         self._prob_reg = self._make_prob_reg(prob_nc)
#
#         # self._set_requires_grads(self.features, requires_grads=False)
#         # self._set_requires_grads(self.classifier, requires_grads=False)
#         self._initialize_weights()
#
#     def _make_conv_features(self, batch_norm=True):
#         cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']  # VGG11
#         layers = []
#         in_channels = 3
#         for v in cfg:
#             if v == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#                 if batch_norm:
#                     layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#                 else:
#                     layers += [conv2d, nn.ReLU(inplace=True)]
#                 in_channels = v
#         return nn.Sequential(*layers)
#
#     def _make_lin_features(self):
#         return nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#         )
#
#     def _make_bb_reg(self, bb_nc):
#         return nn.Sequential(
#             nn.Linear(4096, bb_nc)
#         )
#
#     def _make_prob_reg(self, prob_nc):
#         return nn.Sequential(
#             nn.Linear(4096, prob_nc),
#             nn.Sigmoid()
#         )
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         conv_features = self.features(x)
#         lin_features = self.classifier(conv_features.view(conv_features.size(0), -1))
#         return self._bb_reg(lin_features), self._prob_reg(lin_features)