"""
Import images
"""

import glob
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb, rgb2gray
from skimage.transform import resize
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters
import matplotlib.pyplot as plt


def img_draw(im_list, im_names):
    plt.figure(1)
    n_rows = int(np.sqrt(len(im_list)))
    n_cols = len(im_list) / n_rows
    for img_i in range(len(im_list)):
        plt.subplot(n_cols, n_rows, img_i + 1)
        plt.title(im_names[img_i].split('/')[-1].split('.')[0])
        plt.imshow(im_list[img_i])
    plt.show()


def imp_img(img_name):
    # read
    img = imread(img_name)
    # if gray convert to color
    if len(img.shape) == 2:
        img = gray2rgb(img)
    return img


@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)


@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

n_images = 64
img_size = 40

path = "data"
train_names = sorted(glob.glob(path + "/train/*"))
train_files = []
for i, name_file in enumerate(train_names):
    train_files.append(imp_img(name_file))

train_names = train_names[:n_images]
train_files = train_files[:n_images]

# Resize
for i, img_file in enumerate(train_files):
    train_files[i] = resize(img_file, (img_size, img_size))

img_draw(train_files, train_names)

# Find borders
for i, img_file in enumerate(train_files):
    train_files[i] = sobel_each(img_file)

img_draw(train_files, train_names)

# Chane to gray
for i, img_file in enumerate(train_files):
    train_files[i] = rgb2gray(img_file)

img_draw(train_files, train_names)
