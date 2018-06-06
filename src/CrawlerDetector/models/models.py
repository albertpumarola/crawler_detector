import os
import torch
from collections import OrderedDict


class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, opt):
        model = None

        if model_name == 'object_detector_net_model':
            from .object_detector_net_model import ObjectDetectorNetModel
            model = ObjectDetectorNetModel(opt)
        elif model_name == 'object_detector_net_model_small':
            from .object_detector_net_model_small import ObjectDetectorNetModel
            model = ObjectDetectorNetModel(opt)
        elif model_name == 'object_detector_net_prob_map':
            from .object_detector_net_prob_map import ObjectDetectorNetModel
            model = ObjectDetectorNetModel(opt)
        elif model_name == 'object_detector_net_prob_map2':
            from .object_detector_net_prob_map2 import ObjectDetectorNetModel
            model = ObjectDetectorNetModel(opt)
        elif model_name == 'object_detector_net_prob':
            from .object_detector_net_prob import ObjectDetectorNetModel
            model = ObjectDetectorNetModel(opt)
        else:
            raise ValueError("Model [%s] not recognized." % opt.model)

        print("model [%s] was created" % model.name)
        return model


class BaseModel(object):

    def __init__(self, opt):
        self._name = 'BaseModel'

        self._opt = opt
        self._gpu_ids = opt.gpu_ids
        self._is_train = opt.is_train

        self._device = torch.device("cuda" if self._gpu_ids else "cpu")
        self._save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    @property
    def name(self):
        return self._name

    @property
    def is_train(self):
        return self._is_train

    def set_input(self, input):
        assert False, "set_input not implemented"

    def set_train(self):
        assert False, "set_train not implemented"

    def set_eval(self):
        assert False, "set_eval not implemented"

    def forward(self, keep_data_for_visuals=False):
        assert False, "forward not implemented"

    def get_current_paths(self):
        return OrderedDict()

    def optimize_parameters(self):
        assert False, "optimize_parameters not implemented"

    def get_last_saved_visuals(self):
        return OrderedDict()

    def get_current_errors(self):
        return OrderedDict()

    def get_current_scalars(self):
        return OrderedDict()

    def save(self, label):
        assert False, "save not implemented"

    def load(self):
        assert False, "load not implemented"

    def update_learning_rate(self):
        pass

    def print_network(self, network):
        num_params = 0
        for param in network.parameters():
            num_params += param.numel()
        print(network)
        print('Total number of parameters: %d' % num_params)

    def _save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def _load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
	load_device = 'gpu' if len(self._gpu_ids) > 0 else 'cpu'
        if os.path.exists(load_path):
            optimizer.load_state_dict(torch.load(load_path, map_location=load_device))
            print 'loaded optimizer: %s' % load_path
        else:
            print 'NOT!! loaded optimizer: %s' % load_path

    def _save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    def _load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
	load_device = 'gpu' if len(self._gpu_ids) > 0 else 'cpu'
        if os.path.exists(load_path):
            network.load_state_dict(torch.load(load_path, map_location=load_device))
            print 'loaded net: %s' % load_path
        else:
            print 'NOT!! loaded net: %s' % load_path

    def _move_net_to_gpu(self, net, output_gpu=0):
        if len(self._gpu_ids) > 1:
            return torch.nn.DataParallel(net, device_ids=self._gpu_ids, output_device=output_gpu).to(self._device)
        elif len(self._gpu_ids) == 1:
            return net.to(self._device)
        else:
            return net

    def _update_lr(self, optimizer, old_lr, new_lr, network_name):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print('update %s learning rate: %f -> %f' % (network_name, old_lr, new_lr))
