from PIL import Image, ImageDraw
import numpy as np
import os, sys
from tqdm import tqdm
import argparse
import glob
import xml.etree.ElementTree as ET
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_bbs_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/selected/pos/bbs/', help='Input bbs directory')
parser.add_argument('-o', '--output_bbs_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/selected/pos/', help='Output bbs directory')
args = parser.parse_args()


def read_bb(sample_path):
    '''
    Read last bb
    :param sample_path:
    :return: returns only last bb [u_min, v_min, u_max, v_max]
    '''
    tree = ET.parse(sample_path)
    root = tree.getroot()

    bb = None
    for bndbox in root.findall('./object/bndbox'):
        v_min = int(bndbox.find('xmin').text)
        u_min = int(bndbox.find('ymin').text)
        v_max = int(bndbox.find('xmax').text)
        u_max = int(bndbox.find('ymax').text)
        bb = [u_min, v_min, u_max, v_max]

    return bb

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    if not os.path.isdir(args.output_bbs_dir):
        os.makedirs(args.output_bbs_dir)

    bbs_filepaths = glob.glob(os.path.join(args.input_bbs_dir,'*.xml'))
    bbs_filepaths.sort()

    bbs = {}
    for i, bb_filepath in enumerate(tqdm(bbs_filepaths)):
        bb = read_bb(bb_filepath)
        if bb is not None:
            bbs[os.path.basename(bb_filepath)[:-4]] = bb

    ouput_path = os.path.join(args.output_bbs_dir, 'bbs')
    save_dict(bbs, ouput_path)


if __name__ == '__main__':
    main()