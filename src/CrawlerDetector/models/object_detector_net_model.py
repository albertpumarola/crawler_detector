import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import util.plots as plots
from .models import BaseModel
from networks.networks import NetworksFactory
import numpy as np


class ObjectDetectorNetModel(BaseModel):
    def __init__(self, opt):
        super(ObjectDetectorNetModel, self).__init__(opt)
        self._name = 'ObjectDetectorNetModel'
        self._gpu_bb = 1 if len(self._gpu_ids) > 1 else 0  # everything related bb moved to gpu 1 that has more free mem

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def set_input(self, input):
        # copy images efficiently
        pos_input_img = input['pos_img']
        neg_input_img = input['neg_img']
        self._pos_input_img.resize_(pos_input_img.size()).copy_(pos_input_img)
        self._neg_input_img.resize_(neg_input_img.size()).copy_(neg_input_img)

        # generate gt bb
        pos_gt_norm_bb = input['pos_norm_bb'] if self._is_train else np.array([-1, -1, -1, -1])
        self._pos_input_norm_bb.resize_(pos_gt_norm_bb.size()).copy_(pos_gt_norm_bb)

        # store paths
        self._pos_input_img_path = input['pos_img_path']
        self._neg_input_img_path = input['neg_img_path']

    def forward(self, keep_data_for_visuals=False, keep_estimation=False):
        # clean previous gradients
        if self._is_train:
            if self._opt.lambda_bb > 0:
                self._optimizer_bb.zero_grad()

            if self._opt.lambda_prob > 0:
                self._optimizer_prob.zero_grad()

        # forward pass
        loss_pose = self._forward(keep_data_for_visuals, keep_estimation)

        # compute new gradients
        if self._is_train:
            loss_pose.backward()

    def optimize_parameters(self):
        # optimize net_bb params
        if self._opt.lambda_bb > 0:
            self._optimizer_bb.step()

        # optimize net_prob params
        if self._opt.lambda_prob > 0:
            self._optimizer_prob.step()

    def set_train(self):
        self._net_features.train()
        if self._opt.lambda_bb > 0:
            self._net_bb.train()
        if self._opt.lambda_prob > 0:
            self._net_prob.train()
        self._is_train = True

    def set_eval(self):
        self._net_features.eval()
        if self._opt.lambda_bb > 0:
            self._net_bb.eval()
        if self._opt.lambda_prob > 0:
            self._net_prob.eval()
        self._is_train = False

    def get_current_paths(self):
        return OrderedDict([('pos_img', self._pos_input_img_path),
                            ('neg_img', self._neg_input_img_path)])

    def get_current_errors(self):
        return OrderedDict([('pos_bb_lowres', self._loss_pos_bb_lowres.data[0]),
                            ('pos_prob', self._loss_pos_prob.data[0]),
                            ('neg_prob', self._loss_neg_prob.data[0])])

    def get_current_scalars(self):
        return OrderedDict([('lr_net_bb', self._current_lr_net_bb),
                            ('lr_net_prob', self._current_lr_net_prob)])

    def get_last_saved_estimation(self):
        """
        Returns last model estimation with flag keep_estimation=True
        """
        return self._estim_dict

    def get_last_saved_visuals(self):
        """
        Returns last model visuals with flag keep_data_for_visuals=True
        """
        # visuals return dictionary
        visuals = OrderedDict()

        # estim visuals
        if self._opt.lambda_bb > 0:
            # plot bb
            visuals['pos_gt_bb_gt'] = plots.plot_bb(self._vis_input_pos_img, self._vis_gt_pos_bb_lowres)
            visuals['pos_estim_bb'] = plots.plot_bb(self._vis_input_pos_img, self._vis_estim_pos_bb_lowres,
                                                    label=self._vis_estim_pos_prob,
                                                    display_bb=self._vis_estim_pos_prob>self._opt.classifier_threshold)
            visuals['neg_estim_bb'] = plots.plot_bb(self._vis_input_neg_img, self._vis_estim_neg_bb_lowres,
                                                    label=self._vis_estim_neg_prob,
                                                    display_bb=self._vis_estim_neg_prob>self._opt.classifier_threshold)
        return visuals

    def save(self, label):
        # save networks
        if self._opt.lambda_bb > 0:
            self._save_network(self._net_bb, 'net_bb', label)
        if self._opt.lambda_prob > 0:
            self._save_network(self._net_prob, 'net_prob', label)

        # save optimizers
        if self._opt.lambda_bb > 0:
            self._save_optimizer(self._optimizer_bb, 'net_bb', label)
        if self._opt.lambda_prob > 0:
            self._save_optimizer(self._optimizer_prob, 'net_prob', label)

    def load(self):
        # load networks
        if self._opt.lambda_bb > 0:
            self._load_network(self._net_bb, 'net_bb', self._opt.load_epoch)
        if self._opt.lambda_prob > 0:
            self._load_network(self._net_prob, 'net_prob', self._opt.load_epoch)

        if self._is_train:
            # load optimizers
            if self._opt.lambda_bb > 0:
                self._load_optimizer(self._optimizer_bb, 'net_bb', self._opt.load_epoch)
            if self._opt.lambda_prob > 0:
                self._load_optimizer(self._optimizer_prob, 'net_z', self._opt.load_epoch)

    def update_learning_rate(self):
        # updated learning rate bb net
        if self._opt.lambda_bb > 0:
            lr_decay_net_bb = self._opt.lr_net_bb / self._opt.nepochs_decay
            new_lr_net_bb = self._current_lr_net_bb - lr_decay_net_bb
            self._update_lr(self._optimizer_bb, self._current_lr_net_bb, new_lr_net_bb, 'net_bb')
            self._current_lr_net_bb = new_lr_net_bb

        # update learning rate prob net
        if self._opt.lambda_prob > 0:
            lr_decay_net_prob = self._opt.lr_net_prob / self._opt.nepochs_decay
            new_lr_net_prob = self._current_lr_net_prob - lr_decay_net_prob
            self._update_lr(self._optimizer_prob, self._current_lr_net_prob, new_lr_net_prob, 'net_prob')
            self._current_lr_net_prob = new_lr_net_prob


    # --- INITIALIZER HELPERS ---

    def _init_create_networks(self):
        # features network
        self._net_features = NetworksFactory.get_by_name('vgg_features', freeze=True)
        self._net_features = self._move_net_to_gpu(self._net_features)

        # bb network
        if self._opt.lambda_bb > 0:
            self._net_bb = NetworksFactory.get_by_name('bb_net_values', freeze=False)
            self._net_bb = self._move_net_to_gpu(self._net_bb, output_gpu=self._gpu_bb)

        # prob network
        if self._opt.lambda_prob > 0:
            self._net_prob = NetworksFactory.get_by_name('prob_net', freeze=False)
            self._net_prob = self._move_net_to_gpu(self._net_prob)

    def _init_train_vars(self):
        # initialize learning rate
        self._current_lr_net_bb = self._opt.lr_net_bb
        self._current_lr_net_prob = self._opt.lr_net_prob

        # initialize optimizers
        if self._opt.lambda_bb > 0:
            self._optimizer_bb = torch.optim.Adam(self._net_bb.parameters(), lr=self._current_lr_net_bb,
                                                  weight_decay=self._opt.weight_decay)
        if self._opt.lambda_prob > 0:
            self._optimizer_prob = torch.optim.Adam(self._net_prob.parameters(), lr=self._current_lr_net_prob,
                                                    weight_decay=self._opt.weight_decay)

    def _init_prefetch_inputs(self):
        # prefetch gpu space for images
        self._pos_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._neg_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)

        # prefetch gpu space for poses
        self._pos_input_norm_bb = torch.zeros(self._opt.batch_size, 2, 2).cuda(self._gpu_bb)

    def _init_losses(self):
        # define loss functions
        self._criterion_bb = torch.nn.MSELoss().cuda()  # mean square error
        self._criterion_prob = torch.nn.BCELoss().cuda()  # binary cross-entropy

        # define gt prob
        self._gt_pos_prob = Variable(torch.unsqueeze(torch.ones(self._opt.batch_size), -1)).cuda()
        self._gt_neg_prob = Variable(torch.unsqueeze(torch.zeros(self._opt.batch_size), -1)).cuda()

        # init losses value
        self._loss_pos_bb_lowres = Variable(self._Tensor([0]))
        self._loss_pos_prob = Variable(self._Tensor([0]))
        self._loss_neg_prob = Variable(self._Tensor([0]))

    # --- FORWARD HELPERS ---

    def _forward(self, keep_data_for_visuals, keep_estimation):
        # get data
        pos_imgs = Variable(self._pos_input_img, volatile=(not self._is_train))
        neg_imgs = Variable(self._neg_input_img, volatile=(not self._is_train))
        gt_bb = Variable(self._pos_input_norm_bb, volatile=(not self._is_train))

        # extract features
        pos_features = self._net_features(pos_imgs)
        neg_features = self._net_features(neg_imgs)

        # estimate bb
        estim_pos_bb_lowres = None
        if self._opt.lambda_bb > 0:
            estim_pos_bb_lowres = self._net_bb(pos_features.detach())
            self._loss_pos_bb_lowres = self._criterion_bb(estim_pos_bb_lowres, gt_bb) * self._opt.lambda_bb

        # estimate prob
        estim_pos_prob = None
        estim_neg_prob = None
        if self._opt.lambda_prob > 0:
            estim_pos_prob = self._net_prob(pos_features.detach())
            estim_neg_prob = self._net_prob(neg_features.detach())
            self._loss_pos_prob = self._criterion_prob(estim_pos_prob, self._gt_pos_prob) * self._opt.lambda_prob
            self._loss_neg_prob = self._criterion_prob(estim_neg_prob, self._gt_neg_prob) * self._opt.lambda_prob

        # combined loss (move loss bb to gpu 0)
        total_loss = self._loss_pos_bb_lowres + self._loss_pos_prob + self._loss_neg_prob

        # keep data for visualization
        if keep_data_for_visuals:
            # estimate neg bb for visuals
            estim_neg_bb_lowres = self._net_bb(neg_features.detach()) if self._opt.lambda_bb > 0 else None

            # store visuals
            self._keep_forward_data_for_visualization(pos_imgs, neg_imgs, estim_pos_bb_lowres,
                                                      estim_neg_bb_lowres, gt_bb,
                                                      estim_pos_prob, estim_neg_prob)

        # keep estimated data
        if keep_estimation:
            self._keep_estimation(estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob)

        return total_loss

    def _keep_forward_data_for_visualization(self, pos_imgs, neg_imgs, estim_pos_bb_lowres, estim_neg_bb_lowres,
                                             gt_bb, estim_pos_prob, estim_neg_prob):
        # store img data
        self._vis_input_pos_img = util.tensor2im(pos_imgs.data)
        self._vis_input_neg_img = util.tensor2im(neg_imgs.data)

        # store bb data
        if self._opt.lambda_bb > 0:

            # store bb coords
            self._vis_gt_pos_bb_lowres = self._unormalize_bb(gt_bb.cpu().data[0, ...].numpy())
            self._vis_estim_pos_bb_lowres = self._unormalize_bb(estim_pos_bb_lowres.cpu().data[0, ...].numpy())
            self._vis_estim_neg_bb_lowres = self._unormalize_bb(estim_neg_bb_lowres.cpu().data[0, ...].numpy())

        # store prob data
        if self._opt.lambda_prob > 0:
            self._vis_estim_pos_prob = round(estim_pos_prob.cpu().data[0, ...].numpy(), 2)
            self._vis_estim_neg_prob = round(estim_neg_prob.cpu().data[0, ...].numpy(), 2)
        else:
            self._vis_estim_pos_prob = None
            self._vis_estim_neg_prob = None

    def _keep_estimation(self, estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob):
        self._estim_dict = OrderedDict([('estim_pos_bb_lowres', estim_pos_bb_lowres.cpu().data.numpy()),
                                        ('estim_pos_prob', estim_pos_prob.cpu().data.numpy()),
                                        ('estim_neg_prob', estim_neg_prob.cpu().data.numpy())])

    def _unormalize_bb(self, norm_bb):
        bb = (norm_bb / 2.0 + 0.5) * np.array([self._opt.image_size_h, self._opt.image_size_w,
                                               self._opt.image_size_h, self._opt.image_size_w])
        return bb