from __future__ import absolute_import, division, print_function, unicode_literals


import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

#PATH = os.path.abspath('./drive/My Drive/pix2pix_project')
PATH = os.path.abspath('./')


PATH_IN = PATH + '/input_images_'
PATH_OUT = PATH + '/output_images_'


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255

    return image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image,
        [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return input_image, real_image

def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )

    return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
    input_image = (input_image * 2) - 1
    real_image = (real_image * 2) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)
# 
    if np.random.random() > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def rgb_to_hsv(input_image, real_image):
    return tf.image.rgb_to_hsv(input_image), tf.image.rgb_to_hsv(real_image)

def load_image_train(image_file):
    image_file_in = PATH_IN + image_file
    image_file_out = PATH_OUT + image_file

    input_image, real_image = load(image_file_in), load(image_file_out)
    input_image, real_image = random_jitter(input_image, real_image)
    #input_image, real_image = rgb_to_hsv(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    image_file_in = PATH_IN + image_file
    image_file_out = PATH_OUT + image_file

    input_image, real_image = load(image_file_in), load(image_file_out)
    input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
    #input_image, real_image = rgb_to_hsv(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image
