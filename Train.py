import os
import cv2
import sys
import glob
import time
import random

import numpy as np
import tensorflow as tf

from Define import *
from FCN import *
from FCN_Utils import *

from Utils import *
from Segmentation_Utils import *

os.environ["CUDA_VISIBLE_DEVICES"]="2"

# 1. dataset
TRAIN_PNG_DIRS = ["D:/_ImageDataset/VOC2007/train/png/", "D:/_ImageDataset/VOC2012/png/"]
TEST_PNG_DIRS = ["D:/_ImageDataset/VOC2007/test/png/"]

train_png_paths = []
test_png_paths = []

for train_png_dir in TRAIN_PNG_DIRS:
    train_png_paths += glob.glob(train_png_dir + "*")

for test_png_dir in TEST_PNG_DIRS:
    test_png_paths += glob.glob(test_png_dir + "*")

np.random.shuffle(train_png_paths)
train_png_paths = np.asarray(train_png_paths)

valid_png_paths = list(train_png_paths[:int(len(train_png_paths) * 0.1)])
train_png_paths = list(train_png_paths[int(len(train_png_paths) * 0.1):])

log_print('[i] Train : {}'.format(len(train_png_paths)))
log_print('[i] Valid : {}'.format(len(valid_png_paths)))
log_print('[i] Test : {}'.format(len(test_png_paths)))

# 2. build
color_dic, color_image = get_color_map_dic(DATA_OPTION)

input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, SEGMENT_HEIGHT, SEGMENT_WIDTH, CLASSES])
is_training = tf.placeholder(tf.bool)

logits, prediction_op = FCN_UNet(input_var, is_training)

reshape_logits = tf.reshape(logits, (-1, CLASSES))
reshape_label_var = tf.reshape(label_var, (-1, CLASSES))

class_loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = reshape_logits, labels = reshape_label_var))

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY

loss_op = class_loss_op + l2_reg_loss_op

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # train_op = tf.train.MomentumOptimizer(learning_rate_var, momentum = 0.9).minimize(loss_op)
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#'''
vgg_vars = []
for var in vars:
    if 'vgg_16' in var.name:
        vgg_vars.append(var)

restore_saver = tf.train.Saver(var_list = vgg_vars)
restore_saver.restore(sess, './vgg_16/vgg_16.ckpt')
log_print('[i] restored imagenet parameters (VGG16)')
#'''

saver = tf.train.Saver()
# saver.save(sess, './model/FCN_8000.ckpt')

decay_iterations = np.asarray([0.5, 0.75])
decay_iterations *= MAX_ITERATION
decay_iterations = decay_iterations.astype(np.int32)

learning_rate = INIT_LEARNING_RATE
train_iteration = len(train_png_paths) // BATCH_SIZE
valid_iteration = len(valid_png_paths) // BATCH_SIZE

best_valid_meanIU = 0.0

train_st_time = time.time()
loss_list = []
class_loss_list = []
l2_reg_loss_list = []

for iter in range(1, MAX_ITERATION):

    if iter in decay_iterations:
        learning_rate /= 10
        log_print('[i] learning rate decay : {} -> {}'.format(learning_rate * 10, learning_rate))

    batch_png_paths = random.sample(train_png_paths, BATCH_SIZE)
    batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    batch_label_data = np.zeros((BATCH_SIZE, SEGMENT_HEIGHT, SEGMENT_WIDTH, CLASSES))

    for index, png_path in enumerate(batch_png_paths):
        image_path = png_to_jpg(png_path)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
        mask_image = cv2.imread(png_path)
        mask_image = cv2.resize(mask_image, (SEGMENT_WIDTH, SEGMENT_HEIGHT))
        label = Encode(mask_image, color_dic, CLASSES)

        batch_image_data[index] = image.copy()
        batch_label_data[index] = label.copy()

    _, loss, class_loss, l2_reg_loss = sess.run([train_op, loss_op, class_loss_op, l2_reg_loss_op], feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : True, learning_rate_var : learning_rate})

    loss_list.append(loss)
    class_loss_list.append(class_loss)
    l2_reg_loss_list.append(l2_reg_loss)
        
    if iter % LOG_ITERATION == 0:
        train_time = time.time() - train_st_time
        loss = np.mean(loss_list)
        class_loss = np.mean(class_loss_list)
        l2_reg_loss = np.mean(l2_reg_loss_list)

        log_print('[i] iter : {}, loss : {:.4f}, class_loss = {:.4f}, l2_reg_loss : {:.4f}, time : {}sec'.format(iter, loss, class_loss, l2_reg_loss, int(train_time)))

        train_st_time = time.time()
        loss_list = []
        class_loss_list = []
        l2_reg_loss_list = []

    if iter % VALID_ITERATION == 0:
        # debug train set
        for png_path in train_png_paths[:TRAIN_SAMPLE]:
            image_path = png_to_jpg(png_path)

            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            
            mask_image = cv2.imread(png_path)
            mask_image = cv2.resize(mask_image, (SEGMENT_WIDTH, SEGMENT_HEIGHT))

            gt_image = Encode(mask_image, color_dic, CLASSES)
            decode_gt_image = Decode(gt_image, color_dic)
            
            pred_image = sess.run(prediction_op, feed_dict = {input_var : [image], is_training : False})[0]
            decode_pred_image = Decode(pred_image, color_dic)

            # prediction
            debug_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH * 3, IMAGE_CHANNEL), dtype = np.uint8)
            
            debug_image[:, :IMAGE_WIDTH, :] = image
            debug_image[:, IMAGE_WIDTH : IMAGE_WIDTH * 2, :] = decode_gt_image
            debug_image[:, IMAGE_WIDTH * 2 :, :] = decode_pred_image

            cv2.imwrite('./train_results/iter = {}, {}'.format(iter, os.path.basename(image_path)), debug_image)

        # calculate valid meanIU
        valid_meanIU_list = []

        for index in range(valid_iteration):
            batch_png_paths = valid_png_paths[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
            batch_label_data = np.zeros((BATCH_SIZE, SEGMENT_HEIGHT, SEGMENT_WIDTH, CLASSES))

            for index, png_path in enumerate(batch_png_paths):
                image_path = png_to_jpg(png_path)

                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                
                mask_image = cv2.imread(png_path)
                mask_image = cv2.resize(mask_image, (SEGMENT_WIDTH, SEGMENT_HEIGHT))
                label = Encode(mask_image, color_dic, CLASSES)

                batch_image_data[index] = image.copy()
                batch_label_data[index] = label.copy()

            batch_pred_data = sess.run(prediction_op, feed_dict = {input_var : batch_image_data, is_training : False})
            for pred_image, gt_image in zip(batch_pred_data, batch_label_data):
                valid_meanIU = Calculate_meanIU(pred_image, gt_image)
                valid_meanIU_list.append(valid_meanIU)

        valid_meanIU = np.mean(valid_meanIU_list) * 100
        if best_valid_meanIU < valid_meanIU:
            best_valid_meanIU = valid_meanIU
            saver.save(sess, './model/FCN_{}.ckpt'.format(iter))

        log_print('[i] valid meanIU : {:.2f}%, best valid meanIU : {:.2f}%'.format(valid_meanIU, best_valid_meanIU))
