import torch.nn as nn
import functools
import torch

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'vgg_features':
            from .vgg_features import VggFeatures
            network = VggFeatures(*args, **kwargs)
        elif network_name == 'bb_net':
            from .bb_net import BBNet
            network = BBNet(*args, **kwargs)
        elif network_name == 'bb_net_values':
            from .bb_net_values import BBNetValues
            network = BBNetValues(*args, **kwargs)
        elif network_name == 'prob_net':
            from .prob_net import ProbNet
            network = ProbNet(*args, **kwargs)
        elif network_name == 'small_net':
            from .small_net import SmallNet
            network = SmallNet(*args, **kwargs)
        elif network_name == 'vgg_finetune':
            from .vgg_finetune import VGG11
            network = VGG11(*args, **kwargs)
        elif network_name == 'prob_map_net':
            from .prob_map_net import ProbMapNet
            network = ProbMapNet(*args, **kwargs)
        elif network_name == 'prob_map_net2':
            from .prob_map_net2 import ProbMapNet
            network = ProbMapNet(*args, **kwargs)
        elif network_name == 'prob_map_net3':
            from .prob_map_net3 import ProbMapNet
            network = ProbMapNet(*args, **kwargs)
        elif network_name == 'heat_map_net_prob':
            from .heat_map_net_prob import HeatMapNetProb
            network = HeatMapNetProb(*args, **kwargs)
        elif network_name == 'heat_map_net_prob2':
            from .heat_map_net_prob2 import HeatMapNetProb
            network = HeatMapNetProb(*args, **kwargs)
        elif network_name == 'heat_map_net_prob3':
            from .heat_map_net_prob3 import HeatMapNetProb
            network = HeatMapNetProb(*args, **kwargs)
        elif network_name == 'uv_prob_net':
            from .uv_prob_net import uvProbNet
            network = uvProbNet(*args, **kwargs)
        elif network_name == 'uv_prob_net2':
            from .uv_prob_net2 import uvProbNet
            network = uvProbNet(*args, **kwargs)
        else:
            network = None
            raise ValueError("Network [%s] not recognized." % network_name)

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _set_requires_grads(self, net, requires_grads=True):
        if not requires_grads:
            for param in net.parameters():
                param.requires_grad = False