
import numpy as np
import tensorflow as tf

import VGG16 as vgg
from Define import *

init_fn = tf.contrib.layers.xavier_initializer()

def FCN_UNet(input_var, is_training):
    x = input_var - VGG_MEAN

    i_x = tf.layers.conv2d(inputs = x, filters = 64, kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'i_conv')
    i_x = tf.layers.batch_normalization(inputs = i_x, training = is_training, name = 'i_bn')
    i_x = tf.nn.relu(i_x, name = 'i_relu')
    
    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        feature_maps = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
    
    x = feature_maps[4]
    for i in range(len(feature_maps) - 1):
        prior_feature_map = feature_maps[3 - i]
        shape = prior_feature_map.get_shape().as_list()

        x = tf.layers.conv2d_transpose(inputs = x, filters = shape[-1], kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = init_fn, name = 'up_conv2d_{}'.format(i))
        x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'up_bn_{}'.format(i))
        x = tf.nn.relu(x, name = 'up_relu_{}'.format(i))

        x = tf.concat((x, prior_feature_map), axis = -1)
        
        x = tf.layers.conv2d(inputs = x, filters = shape[-1], kernel_size = [3, 3], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'conv_{}'.format(i))
        x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn_{}'.format(i))
        x = tf.nn.relu(x, name = 'relu_{}'.format(i))

    x = tf.layers.conv2d_transpose(inputs = x, filters = shape[-1], kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = init_fn, name = 'up_conv2d')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'up_bn')
    x = tf.nn.relu(x, name = 'up_relu')

    x = tf.concat((i_x, x), axis = -1)

    x = tf.layers.conv2d(inputs = x, filters = CLASSES, kernel_size = [1, 1], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'conv2d')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

    # predictions = tf.nn.softmax(x, axis = -1, name = 'predictions')
    predictions = x

    return predictions

def FCN(x, is_training, dropout_rate = 0.25):
    x -= VGG_MEAN

    with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
        feature_maps = vgg.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
    
    x = feature_maps[4] # pool5 = 7x7x512

    x = tf.layers.conv2d(inputs = x, filters = 4096, kernel_size = [1, 1], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'conv_6')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn_6')
    x = tf.nn.relu(x, name = 'relu_6')
    x = tf.layers.dropout(x, dropout_rate, training = is_training, name = 'dropout_6')

    x = tf.layers.conv2d(inputs = x, filters = 4096, kernel_size = [1, 1], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'conv_7')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn_7')
    x = tf.nn.relu(x, name = 'relu_7')
    x = tf.layers.dropout(x, dropout_rate, training = is_training, name = 'dropout_7')
    # print(x)

    shape = feature_maps[3].get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs = x, filters = shape[-1], kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = init_fn, name = 'up_conv2d_1')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'up_bn_1')
    x = tf.nn.relu(x, name = 'up_relu_1')
    x += feature_maps[3]
    # print(x)

    shape = feature_maps[2].get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs = x, filters = shape[-1], kernel_size = [3, 3], strides = 2, padding = 'SAME', kernel_initializer = init_fn, name = 'up_conv2d_2')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'up_bn_2')
    x = tf.nn.relu(x, name = 'up_relu_2')
    x += feature_maps[2]
    # print(x)

    shape = feature_maps[1].get_shape().as_list()
    x = tf.layers.conv2d_transpose(inputs = x, filters = shape[-1], kernel_size = [16, 16], strides = 8, padding = 'SAME', kernel_initializer = init_fn, name = 'up_conv2d_3')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'up_bn_3')
    x = tf.nn.relu(x, name = 'up_relu_3')
    # print(x)
    
    x = tf.layers.conv2d(inputs = x, filters = CLASSES, kernel_size = [1, 1], strides = 1, padding = 'SAME', kernel_initializer = init_fn, name = 'conv2d')
    x = tf.layers.batch_normalization(inputs = x, training = is_training, name = 'bn')

    # predictions = tf.nn.softmax(x, axis = -1, name = 'predictions')
    predictions = x

    return predictions

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    segment_outputs = FCN_UNet(input_var, False)

    print(segment_outputs)