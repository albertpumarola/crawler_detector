from __future__ import absolute_import
import sys
sys.path.append('/home/apumarola/code/phd/Crawler-Detector/')

from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from util.tb_visualizer import TBVisualizer
import util.util as util
import util.plots as plots
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np


opt = TrainOptions().parse()
train_data_loader = CustomDatasetDataLoader(opt, is_for_train=True)
train_dataset = train_data_loader.load_data()
train_dataset_size = len(train_data_loader)
print('#training images = %d' % train_dataset_size)

tb_visualizer = TBVisualizer(opt)

for i, data in enumerate(train_dataset):
    pos_img = Variable(data['pos_img'])
    pos_norm_bb = Variable(data['pos_norm_bb'])
    neg_img = Variable(data['neg_img'])
    pos_img_path = data['pos_img_path']
    neg_img_path = data['neg_img_path']

    visuals = OrderedDict()
    img = util.tensor2im(pos_img.data)
    norm_bb = pos_norm_bb.cpu().data[0, :].numpy()

    bb = (norm_bb + 1) / 2.0 * np.array([opt.image_size_h, opt.image_size_w, opt.image_size_h, opt.image_size_w])
    visuals['pos_bb_gt'] = plots.plot_bb(img, bb, 1.00)
    visuals['neg'] = util.tensor2im(neg_img.data)

    tb_visualizer.display_current_results(visuals, i, is_train=True)

    break