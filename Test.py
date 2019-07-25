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

# 1. build
color_dic, color_image = get_color_map_dic(DATA_OPTION)

input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
logits, prediction_op = FCN_UNet(input_var, False)

# 3. test
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './model/FCN_48000.ckpt')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    h, w, c = frame.shape
    tf_image = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
    
    inference_time = time.time()

    prediction = sess.run(prediction_op, feed_dict = {input_var : [tf_image]})[0]

    inference_time = int((time.time() - inference_time) * 1000)
    print('{}ms'.format(inference_time)) # ~13ms

    pred_image = Decode(prediction, color_dic)
    pred_image = cv2.resize(pred_image, (w, h))

    cv2.imshow('Image', frame)
    cv2.imshow('Prediction', pred_image)
    cv2.waitKey(1)
