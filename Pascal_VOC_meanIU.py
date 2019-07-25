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
TEST_PNG_DIRS = ["D:/_ImageDataset/VOC2007/test/png/"]

test_png_paths = []
for test_png_dir in TEST_PNG_DIRS:
    test_png_paths += glob.glob(test_png_dir + "*")

test_length = len(test_png_paths)

# 2. build
color_dic, color_image = get_color_map_dic(DATA_OPTION)

input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
logits, prediction_op = FCN_UNet(input_var, False)

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/FCN_48000.ckpt')

meanIU_list = []

for index, png_path in enumerate(test_png_paths):
    image_path = png_to_jpg(png_path)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
    mask_image = cv2.imread(png_path)
    mask_image = cv2.resize(mask_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    gt_image = Encode(mask_image, color_dic, CLASSES)

    prediction = sess.run(prediction_op, feed_dict = {input_var : [image]})[0]
    pred_image = Decode(prediction, color_dic)

    meanIU = Calculate_meanIU(prediction, gt_image)
    meanIU_list.append(meanIU)

    #cv2.imshow('Image', image)
    #cv2.imshow('Prediction', pred_image)
    #cv2.waitKey(0)

    sys.stdout.write('\r[{}/{}]'.format(index, test_length))
    sys.stdout.flush()

meanIU = np.mean(meanIU_list) * 100
print()
print('[i] test meanIU : {:.2f}%'.format(meanIU))

'''
[i] test meanIU : 86.00%
'''
