from PIL import Image
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-ii', '--input_images_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/neg/imgs', help='Input directory of the images to be cropped')
parser.add_argument('-oi', '--output_images_dir', type=str, default='/home/apumarola/datasets/Dataset-CrawlerDetection/second/processed/neg/imgs', help='Output directory of the images to be cropped')
parser.add_argument('-H', '--new_H', type=int, default=240, help='New image H size')
parser.add_argument('-W', '--new_W', type=int, default=320, help='New image W size')
args = parser.parse_args()

def resize(input_dir, filenames, output_dir):
    for idx, item in enumerate(tqdm(filenames)):
        input_image_path = os.path.join(input_dir, item)
        if os.path.isfile(input_image_path):
            im = Image.open(input_image_path)

            im_reduced = im.resize((args.new_W, args.new_H), Image.ANTIALIAS)

            output_path = os.path.join(output_dir, item)
            output_extension = (os.path.splitext(item))[1][1:]
            im_reduced.save(output_path, output_extension, quality=100)

def main():
    if not os.path.isdir(args.output_images_dir):
        os.makedirs(args.output_images_dir)

    images_filenames = os.listdir(args.input_images_dir)
    images_filenames.sort()
    resize(args.input_images_dir, images_filenames, args.output_images_dir)

if __name__ == '__main__':
    main()