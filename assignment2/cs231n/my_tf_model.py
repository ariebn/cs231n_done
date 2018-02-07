import numpy as np
import tensorflow as tf
# define model

def my_model1(X, y, is_training):
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
def my_model2(X, y, is_training):
    # define our weights (e.g. init_two_layer_convnet)
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    mean1 = tf.get_variable("mean1", shape=[32, ])
    var1 = tf.get_variable("var1", shape=[32, ])
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    mean2 = tf.get_variable("mean2", shape=[64, ])
    var2 = tf.get_variable("var2", shape=[64, ])
    W1 = tf.get_variable("W1", shape=[7744, 1024])
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

    # now add another conv layer to play with ...
    a2 = tf.nn.conv2d(h1p, Wconv2, strides=[1, 1, 1, 1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    # Spatial batch normalization
    h2b = tf.layers.batch_normalization(h2, training=is_training)
    # Max pooling (null, since kernel and stride = 1)
    h2p = tf.nn.max_pool(h2b, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    h2p_flat = tf.reshape(h2p, [-1, 7744])
    # Affine
    X1 = tf.matmul(h2p_flat, W1) + b1
    # Relu
    X1R = tf.nn.relu(X1)
    # Affine
    y_out = tf.matmul(X1R, W2) + b2
    return y_out

def my_model3(X, y, is_training):
    # define our weights (e.g. init_two_layer_convnet)
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    mean1 = tf.get_variable("mean1", shape=[32, ])
    var1 = tf.get_variable("var1", shape=[32, ])
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    mean2 = tf.get_variable("mean2", shape=[64, ])
    var2 = tf.get_variable("var2", shape=[64, ])
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 64])
    bconv3 = tf.get_variable("bconv3", shape=[64])
    mean3 = tf.get_variable("mean3", shape=[64, ])
    var3 = tf.get_variable("var3", shape=[64, ])
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 64, 128])
    bconv4 = tf.get_variable("bconv4", shape=[128])
    mean4 = tf.get_variable("mean4", shape=[128, ])
    var4 = tf.get_variable("var4", shape=[128, ])

    W1 = tf.get_variable("W1", shape=[6272, 2048])
    b1 = tf.get_variable("b1", shape=[2048])
    W2 = tf.get_variable("W2", shape=[2048, 2048])
    b2 = tf.get_variable("b2", shape=[2048])
    W3 = tf.get_variable("W3", shape=[2048, 10])
    b3 = tf.get_variable("b3", shape=[10])

    offset = np.zeros_like(mean1)
    scale = np.ones_like(var1)

    # define our graph (
    # convolution
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + bconv1
    # Relu
    h1 = tf.nn.relu(a1)
    # Spatial batch normalization
    h1b = tf.layers.batch_normalization(h1, training=is_training)
    # Max pooling
    h1p = tf.nn.max_pool(h1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #  dropout
    h1pd = tf.layers.dropout(h1p, rate=0.5,training=is_training)
    #  add another conv layer
    a2 = tf.nn.conv2d(h1pd, Wconv2, strides=[1, 1, 1, 1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    # Spatial batch normalization
    h2b = tf.layers.batch_normalization(h2, training=is_training)
    # yet another conv layer
    a3 = tf.nn.conv2d(h2b, Wconv3, strides=[1, 1, 1, 1], padding='VALID') + bconv3
    h3 = tf.nn.relu(a3)
    # Spatial batch normalization
    h3b = tf.layers.batch_normalization(h3, training=is_training)
    #  dropout
    h3bd = tf.layers.dropout(h3b, rate=0.5, training=is_training)
    #  another discrete conv layer
    a4 = tf.nn.conv2d(h3bd, Wconv4, strides=[1, 1, 1, 1], padding='VALID') + bconv4
    h4 = tf.nn.relu(a4)
    # Spatial batch normalization
    h4b = tf.layers.batch_normalization(h4, training=is_training)
    h4p_flat = tf.reshape(h4b, [-1, 6272])
    # Affine
    X1 = tf.matmul(h4p_flat, W1) + b1
    # Relu
    X1R = tf.nn.relu(X1)
    # Affine
    X2 = tf.matmul(X1R, W2) + b2
    # Relu
    X2R = tf.nn.relu(X2)
    # Affine
    y_out = tf.matmul(X2R, W3) + b3
    return y_out

def my_model4(X, y, is_training):
    # define our weights
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 64])
    bconv2 = tf.get_variable("bconv2", shape=[64])
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 64, 64])
    bconv3 = tf.get_variable("bconv3", shape=[64])
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 64, 128])
    bconv4 = tf.get_variable("bconv4", shape=[128])

    W1 = tf.get_variable("W1", shape=[6272, 2048])
    b1 = tf.get_variable("b1", shape=[2048])
    W2 = tf.get_variable("W2", shape=[2048, 2048])
    b2 = tf.get_variable("b2", shape=[2048])
    W3 = tf.get_variable("W3", shape=[2048, 10])
    b3 = tf.get_variable("b3", shape=[10])

    convl_count = 60                # total rn convolutions
    rblock_size = 2                     #how many convolutions in each building block
    dropout_conv = 0.1
    dropout_affine = 0.5

    # define the graph
    # convolution
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1, 1, 1, 1], padding='VALID') + bconv1
    # Relu
    h1 = tf.nn.relu(a1)
    # Spatial batch normalization
    h1b = tf.layers.batch_normalization(h1, training=is_training)
    # Max pooling
    h1p = tf.nn.max_pool(h1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #  dropout
    h1pd = tf.layers.dropout(h1p, rate=dropout_conv,training=is_training)
    #  another conv layer
    a2 = tf.nn.conv2d(h1pd, Wconv2, strides=[1, 1, 1, 1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    # Spatial batch normalization
    h2b = tf.layers.batch_normalization(h2, training=is_training)
    #  another conv layer ...
    a3 = tf.nn.conv2d(h2b, Wconv3, strides=[1, 1, 1, 1], padding='VALID') + bconv3
    h3 = tf.nn.relu(a3)
    # Spatial batch normalization
    h3b = tf.layers.batch_normalization(h3, training=is_training)
    #  dropout
    h3bd = tf.layers.dropout(h3b, rate=dropout_conv, training=is_training)# yet another conv layer ...
    a4 = tf.nn.conv2d(h3bd, Wconv4, strides=[1, 1, 1, 1], padding='VALID') + bconv4
    h4 = tf.nn.relu(a4)
    # Spatial batch normalization
    h4b = tf.layers.batch_normalization(h4, training=is_training)
    #  dropout
    h4bd = tf.layers.dropout(h4b, rate=dropout_conv, training=is_training)
    # end of initial pipe (pre Resnet)
    xi = h4bd

    # Resnet-like skip feedfroward net
    for i in range (int(convl_count / rblock_size)):
        for j in range (rblock_size):
            xi_block = xi
            with tf.variable_scope('conv_%d' % (i*rblock_size + j)):
                W = tf.get_variable("W", shape=[3, 3, 128, 128])
                b = tf.get_variable("b", shape=[128])
                a = tf.nn.conv2d(xi, W, strides=[1, 1, 1, 1], padding='SAME') + b
                h = tf.nn.relu(a)
                xip = tf.layers.batch_normalization(h, training=is_training)
                xi = tf.layers.dropout(xip, rate=dropout_conv, training=is_training)
            xi = xi + xi_block

    h4p_flat = tf.reshape(xi, [-1, 6272])
    # Affine
    X1 = tf.matmul(h4p_flat, W1) + b1
    # Relu
    X1R = tf.nn.relu(X1)
    #  dropout
    X1Rd = tf.layers.dropout(X1R, rate=dropout_affine, training=is_training)
    # Affine
    X2 = tf.matmul(X1Rd, W2) + b2
    # Relu
    X2R = tf.nn.relu(X2)
    # Try a dropout
    X2Rd = tf.layers.dropout(X1Rd, rate=dropout_affine, training=is_training)
    # Affine
    y_out = tf.matmul(X2Rd, W3) + b3
    return y_out
