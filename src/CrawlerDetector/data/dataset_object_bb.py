import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import cv2
import random
from collections import OrderedDict
import numpy as np
import time
import pickle


class ObjectBBDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(ObjectBBDataset, self).__init__(opt)
        self._name = 'ObjectBBDataset'
        self._is_for_train = is_for_train

        # prepare dataset
        self._root = opt.data_dir
        self._read_dataset()

        # dataset info
        self._image_size_h, self._image_size_w = opt.image_size_h, opt.image_size_w
        self._norm_values = np.array([opt.net_image_size, opt.net_image_size, opt.net_image_size, opt.net_image_size])

    def __getitem__(self, index):

        pos_img = None
        pos_bb = None
        while pos_img is None or (pos_bb is None and self._is_for_train):
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]
            pos_img, pos_img_path = self._get_image_by_id(sample_id)
            pos_bb = self._get_bb_by_id(sample_id)

            if pos_img is None:
                print 'error reading %s, skipping sample' % sample_id

        # neg data
        neg_index = random.randint(0, self._neg_dataset_size - 1)
        neg_img, neg_img_path = self._get_image_by_id(neg_index, pos_sample=False)

        # augment data
        pos_img, pos_bb = self._augment_data(pos_img, pos_bb)
        neg_img, _ = self._crop(neg_img, None)

        # transform data
        pos_img = self._transform(pos_img)
        neg_img = self._transform(neg_img)
        pos_norm_bb = self._normalize_bb(pos_bb) if pos_bb is not None else np.array([-1, -1, -1, -1])
        pos_norm_center = self._get_pos_ceneter(pos_norm_bb)

        self._check_bb(pos_norm_bb)

        # pack data
        sample = {'pos_img': pos_img,
                  'pos_norm_bb': pos_norm_bb,  # bb is represented with top-left and bottom-right coords Nx2 (2x2)
                  'pos_norm_center': pos_norm_center,
                  'neg_img': neg_img,
                  'pos_img_path': pos_img_path,
                  'neg_img_path': neg_img_path
                  }
        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        assert os.path.isdir(self._root), '%s is not a valid directory' % self._root

        # set dataset dir
        pos_imgs_dir = os.path.join(self._root, self._opt.pos_file_name, self._opt.images_folder)
        neg_imgs_dir = os.path.join(self._root, self._opt.neg_file_name, self._opt.images_folder)
        pos_bb_file = os.path.join(self._root, self._opt.pos_file_name, self._opt.bbs_filename)

        # read dataset
        pos_imgs_paths = self._get_all_files_in_subfolders(pos_imgs_dir, self._is_image_file)
        self._neg_imgs_paths = self._get_all_files_in_subfolders(neg_imgs_dir, self._is_image_file)
        self._pos_bbs = self._read_bbs_file(pos_bb_file)
        self._pos_imgs_paths = dict(zip([os.path.basename(path)[:-4] for path in pos_imgs_paths], pos_imgs_paths))

        # read ids
        use_ids_filename = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        use_ids_filepath = os.path.join(self._root, use_ids_filename)
        self._ids = self._read_ids(use_ids_filepath)

        # store dataset size
        self._pos_dataset_size = len(self._pos_imgs_paths)
        self._neg_dataset_size = len(self._neg_imgs_paths)
        self._dataset_size = self._ids.shape[0]

    def _read_ids(self, file_path):
        return np.loadtxt(file_path, delimiter='\t', dtype=np.str)

    def _get_image_by_id(self, id, pos_sample=True):
        path = self._pos_imgs_paths[id] if pos_sample else self._neg_imgs_paths[id]
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), path

    def _get_bb_by_id(self, id):
        if id in self._pos_bbs:
            return np.array(self._pos_bbs[id], dtype=np.float32)
        else:
            return None

    def _read_bbs_file(self, path):
        '''
        Read file with all gt bbs
        :param path: File data must have shape dataset_size x 2*num_points (being num_points = 2)
        :return: Bounding Boxes represented with top-left and right-bottom coords (dataset_size x num_points x 2)
        '''
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _create_transform(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._transform = transforms.Compose(transform_list)

    def _normalize_bb(self, bb):
        return (bb / self._opt.net_image_size - 0.5) * 2

    def _augment_data(self, img, bb):
        aug_type = random.choice(['', 'h', 'v', 'hv']) if bb is not None else None
        if aug_type == 'v':
            img = cv2.flip(img, 1)
        elif aug_type == 'h':
            img = cv2.flip(img, 0)
        elif aug_type == 'hv':
            img = cv2.flip(cv2.flip(img, 0), 1)

        if bb is not None:
            top, left, bottom, right = bb
            bb_h, bb_w = bottom-top, right-left
            if aug_type == 'v':
                left = self._image_size_w - left - bb_w
                right = self._image_size_w - right + bb_w
            elif aug_type == 'h':
                top = self._image_size_h - top - bb_h
                bottom = self._image_size_h - bottom + bb_h
            elif aug_type == 'hv':
                left = self._image_size_w - left - bb_w
                right = self._image_size_w - right + bb_w
                top = self._image_size_h - top - bb_h
                bottom = self._image_size_h - bottom + bb_h
            bb = (top, left, bottom, right)

        img, bb = self._crop(img, bb)

        return img, bb

    def _crop(self, img, bb):
        if bb is not None:
            top, left, bottom, right = bb
            min_u, min_v = int(top * 0.75 + bottom * 0.25), int(left * 0.75 + right * 0.25)
            max_u, max_v = int(top * 0.25 + bottom * 0.75), int(left * 0.25 + right * 0.75)
            min_top = max(max_u-self._opt.net_image_size, 0)
            max_top = min(min_u, self._opt.image_size_h-self._opt.net_image_size)

            min_left = max(max_v - self._opt.net_image_size, 0)
            max_left = min(min_v, self._opt.image_size_w - self._opt.net_image_size)

            top_img = int(random.random()*(max_top-min_top)+min_top)
            left_img = int(random.random()*(max_left-min_left)+min_left)
            bb -= np.array([top_img, left_img, top_img, left_img])
            bb = np.clip(bb, 0, self._opt.net_image_size)

        else:
            top_img = int(random.random() * (self._opt.image_size_h-self._opt.net_image_size))
            left_img = int(random.random() * (self._opt.image_size_w-self._opt.net_image_size))

        img = img[top_img:top_img + self._opt.net_image_size, left_img:left_img + self._opt.net_image_size, :]
        return img, bb

    def _check_bb(self, bb):
        top, left, bottom, right = bb
        assert top <= bottom and left <= right, 'Assert bb not correct'

    def _get_pos_ceneter(self, bb):
        top, left, bottom, right = bb
        return np.array([(top+bottom)/2, (left+right)/2]).astype(np.float32)
