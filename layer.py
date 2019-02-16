import tensorflow as tf
import numpy as np

"""
	Basic Layers
"""
# create a variable
def create_variable(name, shape, initializer = tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 0.997)
    return tf.get_variable(name, shape = shape, initializer = initializer, regularizer = regularizer)

# convolution layer
def conv_layer(x, name, filt_shape, stride = [1, 1, 1, 1], pad = 'SAME'):
	# filt_shape: [conv_size, conv_size, in_channel, out_channel]
	flit = create_variable(name = name + "_filter", shape = filt_shape)
	return tf.nn.conv2d(x, filter = flit, strides = stride, padding = pad, name = name)

# average pooling layer
def avg_pool_layer(x, name, pooling_size, stride = [1, 2, 2, 1], pad = 'SAME'):
	avg = tf.nn.avg_pool(x, ksize = pooling_size, strides = stride, padding = pad)
	return tf.identity(avg, name = name)

# max pooling layer
def max_pool_layer(x, name, pooling_size, stride = [1, 2, 2, 1], pad = 'SAME'):
	return tf.nn.max_pool(x, ksize = pooling_size, strides = stride, padding = pad, name = name)

# batch normalization layer
def batch_normalization(x, name):
	mean, variance = tf.nn.moments(x, axes = [0, 1, 2], keep_dims = False)
	dimension = x.get_shape().as_list()[-1]
	beta = create_variable(name + "_beta", shape = [dimension], initializer = tf.constant_initializer(1.0, tf.float32))
	gamma = create_variable(name + "_gamma", shape = [dimension], initializer = tf.constant_initializer(1.0, tf.float32))
	return tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001, name = name)

# relu layer
def relu_layer(x):
	return tf.nn.leaky_relu(x, alpha = 0.1)

# flatten layer
def flatten_layer(x):
	shape = x.get_shape().as_list()
	return tf.reshape(x, shape = [shape[0], -1])

# fully connected layer
def fc_layer(x, out_d, name):
	shape = x.get_shape().as_list()
	in_d = shape[-1]
	weight = create_variable(name = name + "_weight", shape = [in_d, out_d], initializer = tf.uniform_unit_scaling_initializer(factor = 1.0))
	bias = create_variable(name = name + "bias", shape = [out_d], initializer = tf.truncated_normal_initializer())
	return tf.nn.xw_plus_b(x, weight, bias, name)

# dropout layer
def dropout_layer(x, name, prob):
	return tf.nn.dropout(x, keep_prob = prob, name = name)

"""
	Network layers
"""
def bn_relu_conv(x, name, in_channel, out_channel, conv_size):
	bn_out = batch_normalization(x, name + "_bn")
	relu_out = relu_layer(bn_out)
	conv_out = conv_layer(relu_out, filt_shape = [conv_size, conv_size, in_channel, out_channel], name = name + "_conv")
	return conv_out

def conv_bn_relu(x, name, in_channel, out_channel, conv_size):
	conv_out = conv_layer(x, filt_shape = [conv_size, conv_size, in_channel, out_channel], name = name + "_conv")
	bn_out = batch_normalization(conv_out, name + "_bn")
	relu_out = relu_layer(bn_out)
	return relu_out

def bn_relu(x, name):
	bn_out = batch_normalization(x, name + "_bn")
	relu_out = relu_layer(bn_out)
	return relu_out

# def basic(x, name, in_channel, out_channel):
# 	block0 = bn_relu_conv(x, name + "_basicblock0", in_channel, out_channel, 3)
# 	block1 = bn_relu_conv(block0, name + "_basicblock1", out_channel, out_channel, 3)

# 	x0 = bn_relu_conv(x, name + "_basicblock_pass", in_channel, out_channel, 3)
# 	return x0 + block1

# def bottleneck(x, name, in_channel, out_channel):
# 	# conv 1x1
# 	block0 = bn_relu_conv(x, name + "_bottleneck0" , in_channel, out_channel, 1)
# 	# conv 3x3
# 	block1 = bn_relu_conv(block0, name + "_bottleneck1", out_channel, out_channel, 3)
# 	# conv 1x1
# 	block2 = bn_relu_conv(block1, name + "_bottleneck2", out_channel, out_channel, 1)

# 	x0 = bn_relu_conv(x, name + "_bottleneck_pass0", in_channel, out_channel, 3)
# 	return x0 + block2

# def basic_wide(x, name, in_channel, out_channel):
# 	# conv 3x3
# 	block0 = bn_relu_conv(x, name + "_basicwide0", in_channel, out_channel, 3)
# 	# conv 3x3
# 	block1 = bn_relu_conv(block0, name + "_basicwide1", out_channel, out_channel, 3)

# 	x0 = bn_relu_conv(x, name + "_basicwide_pass", in_channel, out_channel, 3)
# 	return x0 + block1

# def wide_dropout(x, name, prob, in_channel, out_channel):
# 	# conv 3x3
# 	block0 = bn_relu_conv(x, name + "_widedropout0", in_channel, out_channel, 3)
# 	# dropout
# 	dropout = dropout_layer(block0, name + "_widedropout_dropout", prob)
# 	# conv 3x3
# 	block2 = bn_relu_conv(dropout, name + "_widedropout2", out_channel, out_channel, 3)

# 	x0 = bn_relu_conv(x, name + "_widedropout_pass", in_channel, out_channel, 3)
# 	return x0 + block2