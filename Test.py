import os
import cv2
import sys
import glob
import time

import numpy as np
import tensorflow as tf

from Define import *
from FCN import *
from FCN_Utils import *

from Utils import *
from Segmentation_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 1. dataset
TRAIN_PNG_DIRS = ["D:/DB/VOC2007/train/png/", "D:/DB/VOC2012/png/"]
TEST_PNG_DIRS = ["D:/DB/VOC2007/test/png/"]

train_png_paths = []
test_png_paths = []

for train_png_dir in TRAIN_PNG_DIRS:
    train_png_paths += glob.glob(train_png_dir + "*")

for test_png_dir in TEST_PNG_DIRS:
    test_png_paths += glob.glob(test_png_dir + "*")

np.random.shuffle(train_png_paths)
train_png_paths = np.asarray(train_png_paths)

valid_png_paths = train_png_paths[:int(len(train_png_paths) * 0.1)]
train_png_paths = train_png_paths[int(len(train_png_paths) * 0.1):]

# 2. build
color_dic, color_image = get_color_map_dic(DATA_OPTION)

input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
pred_tensors = FCN(input_var, False)

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for png_path in train_png_paths:
    image_path = png_to_jpg(png_path)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
    mask_image = cv2.imread(png_path)
    mask_image = cv2.resize(mask_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    pred_data = sess.run(pred_tensors, feed_dict = {input_var : [image]})[0]
    pred_image = Decode(pred_data, color_dic)

    cv2.imshow('Prediction', pred_image)
    cv2.waitKey(0)