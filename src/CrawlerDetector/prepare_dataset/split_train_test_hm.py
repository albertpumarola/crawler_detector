from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_dir_imgs', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/pos/imgs', help='Input directory of the images to be cropped')
parser.add_argument('-ib', '--input_dir_hms', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/pos/hms/', help='Input directory of the images to be cropped')
parser.add_argument('-p', '--train_ratio', type=float, default=0.9, help='train ratio samples')
parser.add_argument('-o', '--output_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/', help='Output path')


args = parser.parse_args()

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    imgs_ids = [filename[:-4] for filename in os.listdir(args.input_dir_imgs)]
    hms_ids = set([fname[:-4] for fname in os.listdir(args.input_dir_hms)])

    train = []
    val = []
    for id in imgs_ids:
        if id in hms_ids:
            if np.random.rand() <= args.train_ratio:
                train.append(id)
            else:
                val.append(id)

    print len(train), len(val)

    train_save_dir = os.path.join(args.output_dir, 'train_ids.csv')
    np.savetxt(train_save_dir, np.array(train), delimiter='\t', fmt="%s")

    val_save_dir = os.path.join(args.output_dir, 'test_ids.csv')
    np.savetxt(val_save_dir, np.array(val), delimiter='\t', fmt="%s")




if __name__ == '__main__':
    main()