import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
import cv2
import random
from collections import OrderedDict
import numpy as np
import time
import pickle


class ObjectHMDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(ObjectHMDataset, self).__init__(opt)
        self._name = 'ObjectHMDataset'
        self._is_for_train = is_for_train

        # prepare dataset
        self._root = opt.data_dir
        self._read_dataset()

        # dataset info
        self._image_size_h, self._image_size_w = opt.image_size_h, opt.image_size_w

    def __getitem__(self, index):

        pos_img = None
        pos_hm = None
        while pos_img is None or (pos_hm is None and self._is_for_train):
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]
            pos_img, pos_img_path = self._get_image_by_id(sample_id)
            pos_hm = self._get_hm_by_id(sample_id)

            if pos_img is None:
                print 'error reading %s, skipping sample' % sample_id

        # neg data
        neg_index = random.randint(0, self._neg_dataset_size - 1)
        neg_img, neg_img_path = self._get_image_by_id(neg_index, pos_sample=False)

        # augment data
        pos_img, pos_hm = self._augment_data(pos_img, pos_hm)
        neg_img, _ = self._augment_data(neg_img, None)

        # transform data
        pos_img = self._transform(pos_img)
        neg_img = self._transform(neg_img)
        pos_norm_hm = self._normalize_hm(pos_hm) if pos_hm is not None else np.array([-1, -1], dtype=np.float32)
        # pos_norm_hm = np.array([-1, -1]).astype(np.float32)
        # print pos_norm_hm

        # pack data
        sample = {'pos_img': pos_img,
                  'pos_norm_pose': pos_norm_hm,  # hm pose is represented with ij
                  'neg_img': neg_img,
                  'pos_img_path': pos_img_path,
                  'neg_img_path': neg_img_path
                  }

        if pos_norm_hm.dtype != 'float32':
            print pos_norm_hm.dtype, pos_norm_hm

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset(self):
        assert os.path.isdir(self._root), '%s is not a valid directory' % self._root

        pos_file_name = self._opt.train_pos_file_name if self._is_for_train else self._opt.test_pos_file_name

        # set dataset dir
        pos_imgs_dir = os.path.join(self._root, pos_file_name, self._opt.images_folder)
        neg_imgs_dir = os.path.join(self._root, self._opt.neg_file_name, self._opt.images_folder)
        pos_hm_file = os.path.join(self._root, pos_file_name, self._opt.hms_filename)

        # read dataset
        pos_imgs_paths = self._get_all_files_in_subfolders(pos_imgs_dir, self._is_image_file)
        self._neg_imgs_paths = self._get_all_files_in_subfolders(neg_imgs_dir, self._is_image_file)
        self._pos_hms_dict = self._read_hms_file(pos_hm_file)
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

    def _get_hm_by_id(self, id):
        if id in self._pos_hms_dict:
            return np.array(self._pos_hms_dict[id], dtype=np.int)
        else:
            return None

    def _read_hms_file(self, path):
        '''
        Read file with all gt hms
        :param path: File data must have shape dataset_size x 2*num_points (being num_points = 2)
        :return: Bounding Boxes represented with top-left and right-bottom coords (dataset_size x num_points x 2)
        '''
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _create_transform(self):
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        self._transform = transforms.Compose(transform_list)

    def _normalize_hm(self, hm):
        return (hm.astype(np.float32) / self._opt.net_image_size - 0.5) * 2.

    def _augment_data(self, img, hm):
        aug_type = random.choice(['', 'h', 'v', 'hv'])
        if aug_type == 'v':
            img = cv2.flip(img, 1)
        elif aug_type == 'h':
            img = cv2.flip(img, 0)
        elif aug_type == 'hv':
            img = cv2.flip(cv2.flip(img, 0), 1)

        if hm is not None:
            v, u = hm
            if aug_type == 'v':
                u = self._image_size_w - u
            elif aug_type == 'h':
                v = self._image_size_h - v
            elif aug_type == 'hv':
                v = self._image_size_h - v
                u = self._image_size_w - u
            hm = (v, u)

        img, hm = self._crop(img, hm)

        return img, hm

    def _crop(self, img, hm, min_margin=20):
        if hm is not None:
            i, j = hm
            min_i, min_j = np.clip(i-min_margin, 0, self._opt.image_size_h), np.clip(j-min_margin, 0, self._opt.image_size_w)
            max_i, max_j = np.clip(i+min_margin, 0, self._opt.image_size_h), np.clip(j+min_margin, 0, self._opt.image_size_w)
            min_top = max(max_i-self._opt.net_image_size, 0)
            max_top = min(min_i, self._opt.image_size_h-self._opt.net_image_size)

            min_left = max(max_j - self._opt.net_image_size, 0)
            max_left = min(min_j, self._opt.image_size_w - self._opt.net_image_size)

            top_img = int(random.random()*(max_top-min_top)+min_top)
            left_img = int(random.random()*(max_left-min_left)+min_left)
            hm -= np.array([top_img, left_img])
            hm = np.clip(hm, 0, self._opt.net_image_size)

        else:
            top_img = int(random.random() * (self._opt.image_size_h-self._opt.net_image_size))
            left_img = int(random.random() * (self._opt.image_size_w-self._opt.net_image_size))

        img = img[top_img:top_img + self._opt.net_image_size, left_img:left_img + self._opt.net_image_size, :]
        return img, hm
