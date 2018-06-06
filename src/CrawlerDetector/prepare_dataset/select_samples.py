#!/usr/bin/python
import os
from tqdm import tqdm
import argparse
import glob
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_bags_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/original/neg/imgs/', help='Input images directory')
parser.add_argument('-oi', '--output_bags_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/neg/imgs/', help='Output images directory')
parser.add_argument('-n', '--every_n_frames', type=int, default=3, help='Select every n frames')
args = parser.parse_args()

def main():
    input_bags_dir = glob.glob(os.path.join(args.input_bags_dir, '*/color/'))

    save_dir = args.output_bags_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for bag_dir in tqdm(input_bags_dir):
        images_filenames = os.listdir(bag_dir)
        images_filenames.sort()
        for i, image_filename in enumerate(images_filenames):
            if i % args.every_n_frames == 0:
                src_filename = os.path.join(bag_dir, os.path.basename(image_filename))
                save_filename = os.path.join(save_dir, os.path.basename(image_filename))
                copyfile(src_filename, save_filename)


if __name__ == '__main__':
    main()