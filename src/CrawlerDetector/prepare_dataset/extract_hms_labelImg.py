from PIL import Image, ImageDraw
import numpy as np
import os, sys
from tqdm import tqdm
import argparse
import glob
import xml.etree.ElementTree as ET
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_hms_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/test/hms/', help='Input hms directory')
parser.add_argument('-o', '--output_hms_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/test/', help='Output hms directory')
args = parser.parse_args()


def read_hm(sample_path):
    '''
    Read last hm
    :param sample_path:
    :return: returns only last hm [u_min, v_min, u_max, v_max]
    '''
    tree = ET.parse(sample_path)
    root = tree.getroot()

    hm = None
    for bndbox in root.findall('./object/bndbox'):
        v_min = int(bndbox.find('xmin').text)
        u_min = int(bndbox.find('ymin').text)
        hm = [u_min, v_min]

    return hm

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def main():
    if not os.path.isdir(args.output_hms_dir):
        os.makedirs(args.output_hms_dir)

    hms_filepaths = glob.glob(os.path.join(args.input_hms_dir,'*.xml'))
    hms_filepaths.sort()

    hms = {}
    for i, hm_filepath in enumerate(tqdm(hms_filepaths)):
        hm = read_hm(hm_filepath)
        if hm is not None:
            hms[os.path.basename(hm_filepath)[:-4]] = hm

    ouput_path = os.path.join(args.output_hms_dir, 'hms')
    save_dict(hms, ouput_path)


if __name__ == '__main__':
    main()