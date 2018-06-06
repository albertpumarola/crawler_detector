from __future__ import print_function
from PIL import Image
import numpy as np
import os


def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0):
    if len(img.shape) == 4:
        img = img[idx]

    img = img.cpu().float()

    if unnormalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        for i, m, s in zip(img, mean, std):
            i.mul_(s).add_(m)

    image_numpy = img.numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*255.0

    return image_numpy_t.astype(imtype)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)