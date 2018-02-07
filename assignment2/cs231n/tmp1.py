import numpy as np
import tensorflow as tf
# define model

def my_model1(X, y, is_training):
    print (' forom module *(*(*(')
    # define our weights (e.g. init_two_layer_convnet)
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    mean1 = tf.get_variable("mean1", shape=[32, ])
    var1 = tf.get_variable("var1", shape=[32, ])
    W1 = tf.get_variable("W1", shape=[5408, 1024])
    b1 = tf.get_variable("b1", shape=[1024])
    W2 = tf.get_variable("W2", shape=[1024, 10])
    b2 = tf.get_variable("b2", shape=[10])

    offset = np.zeros_like(mean1)
    scale = np.ones_like(var1)

    # define our graph (e.g. two_layer_convnet)
    # convolution
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + bconv1
    # Relu
    h1 = tf.nn.relu(a1)
    # Spatial batch normalization
    h1b = tf.layers.batch_normalization(h1, training=is_training)
    # Max pooling
    h1p = tf.nn.max_pool(h1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    h1p_flat = tf.reshape(h1p, [-1, 5408])
    # Affine
    X1 = tf.matmul(h1p_flat, W1) + b1
    # Relu
    X1R = tf.nn.relu(X1)
    # Affine
    y_out = tf.matmul(X1R, W2) + b2
    return y_out
#
