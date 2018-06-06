import argparse
import os

try:
    from util import util
except ImportError:
    from CrawlerDetector.util import util

import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self._opt = None

    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/', help='path to train data')
        self._parser.add_argument('--neg_file_name', type=str, default='neg', help='neg images folder')
        self._parser.add_argument('--train_pos_file_name', type=str, default='pos', help='train images folder')
        self._parser.add_argument('--test_pos_file_name', type=str, default='test', help='test images folder')
        self._parser.add_argument('--images_folder', type=str, default='imgs', help='pos images folder')
        self._parser.add_argument('--bbs_filename', type=str, default='bbs.pkl', help='pos bounding boxes files')
        self._parser.add_argument('--hms_filename', type=str, default='hms.pkl', help='pos hm files')
        self._parser.add_argument('--train_ids_file', type=str, default='train_ids.csv',help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='test_ids.csv', help='file containing test ids')

        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
        self._parser.add_argument('--image_size_h', type=int, default=240, help='input image size')
        self._parser.add_argument('--image_size_w', type=int, default=320, help='input image size')
        self._parser.add_argument('--net_image_size', type=int, default=224, help='input image size')
        self._parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--name', type=str, default='pretrained_model', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--dataset_mode', type=str, default='object_hm', help='chooses how datasets are loaded. [object_bb]')
        self._parser.add_argument('--model', type=str, default='object_detector_net_prob', help='chooses which model to use. [object_detector_net_model]')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')

        self._parser.add_argument('--poses_g_sigma', type=float, default=0.06, help='initial learning rate for adam')
        self._parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for small_net adam')
        # self._parser.add_argument('--lr_net_bb', type=float, default=0.0002, help='initial learning rate for net_proj adam')
        # self._parser.add_argument('--lr_net_prob', type=float, default=0.0002, help='initial learning rate for net_z adam')
        self._parser.add_argument('--lambda_bb', type=float, default=1000, help='lambda for bb in loss')
        self._parser.add_argument('--lambda_prob', type=float, default=1000, help='lambda for prob in loss')
        self._parser.add_argument('--classifier_threshold', type=float, default=0.5, help='classifier threshold')

        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt, _ = self._parser.parse_known_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[1]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
