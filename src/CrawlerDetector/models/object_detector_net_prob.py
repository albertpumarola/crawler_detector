import torch
from collections import OrderedDict
import numpy as np
from torchsummary import summary

try:
    import util.util as util
    import util.plots as plots
    from models import BaseModel
    from networks.networks import NetworksFactory
except ImportError:
    import CrawlerDetector.util.util as util
    import CrawlerDetector.util.plots as plots
    from CrawlerDetector.models.models import BaseModel
    from CrawlerDetector.networks.networks import NetworksFactory



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
        if self._is_train:
            self._init_prefetch_inputs()

        # init losses
        if self._is_train:
            self._init_losses()

    def set_input(self, input):
        # copy images efficiently
        pos_input_img = input['pos_img']
        neg_input_img = input['neg_img']
        self._pos_input_img.resize_(pos_input_img.size()).copy_(pos_input_img)
        self._neg_input_img.resize_(neg_input_img.size()).copy_(neg_input_img)

        # gt p
        self._pos_input_p.resize_(input['pos_norm_pose'].size()).copy_(input['pos_norm_pose'])

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

    def test(self, image, do_normalize_output=False):
        with torch.no_grad():
            estim_p, estim_prob = self._net(image)
            uv_max = np.squeeze(self._unormalize_pose(estim_p.detach().numpy()))

        return None, uv_max, estim_prob.detach().numpy()

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
                            ('neg_img', self._neg_input_img_path)
                            ])

    def get_current_errors(self):
        return OrderedDict([('pos_p', self._loss_pos_p.item()),
                            ('pos_prob', self._loss_pos_prob.item()),
                            ('neg_prob', self._loss_neg_prob.item())
                            ])

    def get_current_scalars(self):
        return OrderedDict([('lr', self._current_lr)])

    def get_last_saved_estimation(self):
        """
        Returns last model estimation with flag keep_estimation=True
        """
        return None

    def get_last_saved_visuals(self):
        """
        Returns last model visuals with flag keep_data_for_visuals=True
        """
        # visuals return dictionary
        visuals = OrderedDict()
        visuals['pos_gt_p'] = plots.plot_center(self._vis_input_pos_img, self._vis_gt_pos_p, prob=self._vis_gt_pos_prob)
        visuals['pos_estim_p'] = plots.plot_center(self._vis_input_pos_img, self._vis_estim_pos_p, prob=self._vis_estim_pos_prob)
        visuals['neg_gt_p'] = plots.plot_center(self._vis_input_neg_img, (0, 0), prob=self._vis_gt_neg_prob)
        visuals['neg_estim_p'] = plots.plot_center(self._vis_input_neg_img, self._vis_estim_neg_p, prob=self._vis_estim_neg_prob)
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
        self._net = NetworksFactory.get_by_name('uv_prob_net2', num_nc=32)
        self._net.init_weights()
        self._net = self._move_net_to_gpu(self._net)
        if len(self._gpu_ids) > 0:
            summary(self._net, (3, self._opt.net_image_size, self._opt.net_image_size))
        #     summary(self._net._prob_reg, (3, self._opt.net_image_size, self._opt.net_image_size))

    def _init_train_vars(self):
        # initialize learning rate
        self._current_lr = self._opt.lr

        # initialize optimizers
        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._net.parameters()),
                                           lr=self._current_lr)

    def _init_prefetch_inputs(self):
        # prefetch gpu space for images
        self._pos_input_img = torch.zeros([self._opt.batch_size, 3, self._opt.net_image_size, self._opt.net_image_size]).to(self._device)
        self._neg_input_img = torch.zeros([self._opt.batch_size, 3, self._opt.net_image_size, self._opt.net_image_size]).to(self._device)

        # prefetch gpu space for poses
        self._pos_input_p = torch.zeros([self._opt.batch_size, 2]).to(self._device)

        # prefetch gpu space for prob
        self._gt_pos_prob = torch.unsqueeze(torch.ones(self._opt.batch_size), -1).to(self._device)
        self._gt_neg_prob = torch.unsqueeze(torch.zeros(self._opt.batch_size), -1).to(self._device)

    def _init_losses(self):
        # define loss functions
        self._criterion_pos = torch.nn.SmoothL1Loss().to(self._device)
        # self._criterion_pos = torch.nn.MSELoss().to(self._device)
        self._criterion_prob = torch.nn.BCEWithLogitsLoss().to(self._device)

        # init losses value
        self._loss_pos_p = torch.zeros(1, requires_grad=True).to(self._device)
        self._loss_pos_prob = torch.zeros(1, requires_grad=True).to(self._device)
        self._loss_neg_prob = torch.zeros(1, requires_grad=True).to(self._device)

    # --- FORWARD HELPERS ---

    def _forward(self, keep_data_for_visuals, keep_estimation):
        with torch.set_grad_enabled(self._is_train):
            # estim bb and prob
            pos_p, pos_prob = self._net(self._pos_input_img)
            neg_p, neg_prob = self._net(self._neg_input_img)

            # calculate losses
            self._loss_pos_p = self._criterion_pos(pos_p, self._pos_input_p) * self._opt.lambda_bb
            self._loss_pos_prob = self._criterion_prob(pos_prob, self._gt_pos_prob) * self._opt.lambda_prob
            self._loss_neg_prob = self._criterion_prob(neg_prob, self._gt_neg_prob) * self._opt.lambda_prob

            # combined loss (move loss bb to gpu 0)
            total_loss = self._loss_pos_p + self._loss_pos_prob + self._loss_neg_prob

            # keep data for visualization
            if keep_data_for_visuals:
                # store visuals
                self._keep_forward_data_for_visualization(self._pos_input_img, self._neg_input_img, pos_p, neg_p,
                                                          self._pos_input_p, pos_prob, neg_prob,
                                                          self._gt_pos_prob, self._gt_neg_prob)

            # keep estimated data
            if keep_estimation:
                pass

        return total_loss

    def _keep_forward_data_for_visualization(self, pos_imgs, neg_imgs, pos_p, neg_p, gt_pos_p,
                                             pos_prob, neg_prob, gt_pos_prob, gt_neg_prob):
        # store img data
        self._vis_input_pos_img = util.tensor2im(pos_imgs.detach())
        self._vis_input_neg_img = util.tensor2im(neg_imgs.detach())

        self._vis_gt_pos_p = self._unormalize_pose(gt_pos_p.cpu().detach()[0, ...].numpy())
        self._vis_estim_pos_p = self._unormalize_pose(pos_p.cpu().detach()[0, ...].numpy())
        self._vis_estim_neg_p = self._unormalize_pose(neg_p.cpu().detach()[0, ...].numpy())

        self._vis_gt_pos_prob = round(gt_pos_prob.cpu().data[0, ...].numpy(), 2)
        self._vis_gt_neg_prob = round(gt_neg_prob.cpu().data[0, ...].numpy(), 2)
        self._vis_estim_pos_prob = round(pos_prob.cpu().data[0, ...].numpy(), 2)
        self._vis_estim_neg_prob = round(neg_prob.cpu().data[0, ...].numpy(), 2)

    def _keep_estimation(self, estim_pos_bb_lowres, estim_pos_prob, estim_neg_prob):
        return None

    def _unormalize_pose(self, norm_pose):
        return (norm_pose/2.0 + 0.5) * self._opt.net_image_size
