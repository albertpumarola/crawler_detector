import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import util.plots as plots
from .models import BaseModel
from networks.networks import NetworksFactory
import numpy as np
import torch.utils.model_zoo as model_zoo


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
        # else:
        #     self._load_vgg_weights()

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

        # gt bb
        pos_gt_norm_bb = input['pos_norm_bb']
        self._pos_input_norm_bb.resize_(pos_gt_norm_bb.size()).copy_(pos_gt_norm_bb)

        # store paths
        self._pos_input_img_path = input['pos_img_path']
        self._neg_input_img_path = input['neg_img_path']

    def forward(self, keep_data_for_visuals=False, keep_estimation=False):
        # clean previous gradients
        if self._is_train:
           self._optimizer.zero_grad()

        # forward pass
        loss_pose = self._forward(keep_data_for_visuals, keep_estimation)

        # compute new gradients
        if self._is_train:
            loss_pose.backward()

    def test(self, image):
        # bb as (top, left, bottom, right)
        estim_bb_lowres, estim_prob = self._net(Variable(image, volatile=True))
        bb = self._unormalize_bb(estim_bb_lowres.data.numpy()).astype(np.int)
        prob = estim_prob.data.numpy()
        return bb, prob

    def optimize_parameters(self):
        self._optimizer.step()

    def set_train(self):
        self._net.train()
        self._is_train = True

    def set_eval(self):
        self._net.eval()
        self._is_train = False

    def get_current_paths(self):
        return OrderedDict([('pos_img', self._pos_input_img_path),
                            ('neg_img', self._neg_input_img_path)])

    def get_current_errors(self):
        return OrderedDict([('pos_bb_lowres', self._loss_pos_bb_lowres.data[0]),
                            ('pos_prob', self._loss_pos_prob.data[0]),
                            ('neg_prob', self._loss_neg_prob.data[0])])

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

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
                                                    display_bb=True)
            visuals['neg_estim_bb'] = plots.plot_bb(self._vis_input_neg_img, self._vis_estim_neg_bb_lowres,
                                                    label=self._vis_estim_neg_prob,
                                                    display_bb=True)
        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._net, 'net', label)

        # save optimizers
        self._save_optimizer(self._optimizer, 'net', label)

    def load(self):
        # load networks
        self._load_network(self._net, 'net', self._opt.load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer, 'net', self._opt.load_epoch)

    def update_learning_rate(self):
        # updated learning rate bb net
        lr_decay= self._opt.lr / self._opt.nepochs_decay
        new_lr = self._current_lr - lr_decay
        self._update_lr(self._optimizer, self._current_lr, new_lr, 'net')
        self._current_lr = new_lr


    # --- INITIALIZER HELPERS ---

    def _init_create_networks(self):
        # features network
        # self._net = NetworksFactory.get_by_name('small_net', freeze=False)
        self._net = NetworksFactory.get_by_name('vgg_finetune')
        self._net = self._move_net_to_gpu(self._net)

    def _init_train_vars(self):
        # initialize learning rate
        self._current_lr = self._opt.lr

        # initialize optimizers
        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                           lr=self._current_lr)

    def _init_prefetch_inputs(self):
        # prefetch gpu space for images
        self._pos_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size_h, self._opt.image_size_w)
        self._neg_input_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size_h, self._opt.image_size_w)

        # prefetch gpu space for poses
        self._pos_input_norm_bb = torch.zeros(self._opt.batch_size, 2, 2).cuda(self._gpu_bb)

    def _init_losses(self):
        # define loss functions
        self._criterion_bb = torch.nn.SmoothL1Loss().cuda()  # mean square error
        self._criterion_prob = torch.nn.BCELoss().cuda()  # binary cross-entropy
        # self._criterion_prob = torch.nn.MSELoss().cuda()  # mean square error

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

        # estim bb and prob
        estim_pos_bb_lowres, estim_pos_prob = self._net(pos_imgs)
        estim_neg_bb_lowres, estim_neg_prob = self._net(neg_imgs)

        # calculate losses
        self._loss_pos_bb_lowres = self._criterion_bb(estim_pos_bb_lowres, gt_bb) * self._opt.lambda_bb
        self._loss_pos_prob = self._criterion_prob(estim_pos_prob, self._gt_pos_prob) * self._opt.lambda_prob
        self._loss_neg_prob = self._criterion_prob(estim_neg_prob, self._gt_neg_prob) * self._opt.lambda_prob

        # combined loss (move loss bb to gpu 0)
        total_loss = self._loss_pos_bb_lowres + self._loss_pos_prob + self._loss_neg_prob

        # keep data for visualization
        if keep_data_for_visuals:
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
        # bb = (norm_bb / 2.0 + 0.5) * np.array([self._opt.net_image_size, self._opt.net_image_size,
        #                                        self._opt.net_image_size, self._opt.net_image_size])
        bb = norm_bb * self._opt.net_image_size
        return bb

    # def _load_vgg_weights(self):
    #     pass
    #     # self._net.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg11_bn-6002323d.pth'),
    #     #                           strict=False)
