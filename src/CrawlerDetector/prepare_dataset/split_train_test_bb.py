from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_dir_imgs', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/selected/pos/imgs', help='Input directory of the images to be cropped')
parser.add_argument('-ib', '--input_dir_bbs', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/selected/pos/bbs/', help='Input directory of the images to be cropped')
parser.add_argument('-o', '--output_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/selected/', help='Output path')
args = parser.parse_args()

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    images_filenames = os.listdir(args.input_dir_imgs)
    bbs_filenames = set([fname[:-4] for fname in os.listdir(args.input_dir_bbs)])

    train = []
    val = []
    for image_filename in images_filenames:
        image_id = image_filename[:-4]
        if image_id in bbs_filenames:
            train.append(image_id)
        else:
            val.append(image_id)

    train_save_dir = os.path.join(args.output_dir, 'train_ids.csv')
    np.savetxt(train_save_dir, np.array(train), delimiter='\t', fmt="%s")

    val_save_dir = os.path.join(args.output_dir, 'test_ids.csv')
    np.savetxt(val_save_dir, np.array(val), delimiter='\t', fmt="%s")




if __name__ == '__main__':
    main()