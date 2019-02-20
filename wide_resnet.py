import tensorflow as tf
import numpy as np
import layer

# def build_wide_resnet(x, k, N, num_classes):
#     channels = [3, 16, 16 * k, 32 * k, 64 * k]
    
#     # conv1
#     conv1 = layer.conv_layer(x, name = "conv1", filt_shape = [3, 3, channels[0], channels[1]])
#     bn1 = layer.batch_normalization(conv1, name = "bn1")
#     relu1 = layer.relu_layer(bn1)
    
#     # conv2
#     conv20 = layer.conv_layer(relu1, name = "conv20", filt_shape = [3, 3, channels[1], channels[2]])
#     bn20 = layer.batch_normalization(conv20, name = "bn20")
#     relu20 = layer.relu_layer(bn20)
#     conv21 = layer.conv_layer(relu20, name = "conv21", filt_shape = [3, 3, channels[2], channels[2]])
#     conv20_block = layer.conv_layer(relu1, name = "conv2_block", filt_shape = [3, 3, channels[1], channels[2]])
#     output21 = conv20_block + relu20

#     bn22 = layer.batch_normalization(output21, name = "bn22")
#     relu22 = layer.relu_layer(bn22)
#     conv23 = layer.conv_layer(relu22, name = "conv23", filt_shape = [3, 3, channels[2], channels[2]])
#     bn23 = layer.batch_normalization(conv23, name = "bn23")
#     relu23 = layer.relu_layer(bn23)
#     conv24 = layer.conv_layer(relu23, name = "conv24", filt_shape = [3, 3, channels[2], channels[2]])
#     output24 = output21 + conv24

#     # downsampling
#     downsampling0 = layer.avg_pool_layer(output24, name = "downsampling0", pooling_size = [1, 2, 2, 1])

#     # conv3
#     bn30 = layer.batch_normalization(downsampling0, name = "bn30")
#     relu30 = layer.relu_layer(bn30)

#     conv31 = layer.conv_layer(relu30, name = "conv31", filt_shape = [3, 3, channels[2], channels[3]])
#     bn31 = layer.batch_normalization(conv31, name = "bn31")
#     relu31 = layer.relu_layer(bn31)
#     conv32 = layer.conv_layer(relu31, name = "conv32", filt_shape = [3, 3, channels[3], channels[3]])
#     conv31_block = layer.conv_layer(relu30, name = "conv31_block", filt_shape = [3, 3, channels[2], channels[3]])
#     output31 = conv31_block + conv32

#     bn32 = layer.batch_normalization(output31, name = "bn32")
#     relu32 = layer.relu_layer(bn32)
#     conv33 = layer.conv_layer(relu32, name = "conv33", filt_shape = [3, 3, channels[3], channels[3]])
#     bn33 = layer.batch_normalization(conv33, name = "bn33")
#     relu33 = layer.relu_layer(bn33)
#     conv34 = layer.conv_layer(relu33, name = "conv34", filt_shape = [3, 3, channels[3], channels[3]])
#     output34 = conv34 + output31

#     # downsampling
#     downsampling1 = layer.avg_pool_layer(output34, name = "downsampling1", pooling_size = [1, 2, 2, 1])

#     # conv4
#     bn40 = layer.batch_normalization(downsampling1, name = "bn40")
#     relu40 = layer.relu_layer(bn40)

#     conv41 = layer.conv_layer(relu40, name = "conv41", filt_shape = [3, 3, channels[3], channels[4]])
#     bn41 = layer.batch_normalization(conv41, name = "conv41")
#     relu41 = layer.relu_layer(bn41)
#     conv42 = layer.conv_layer(relu41, name = "conv42", filt_shape = [3, 3, channels[4], channels[4]])
#     conv41_block = layer.conv_layer(relu40, name = "conv41_block", filt_shape = [3, 3, channels[3], channels[4]])
#     output41 = conv41_block + conv42

#     bn42 = layer.batch_normalization(output41, name = "bn42")
#     relu42 = layer.relu_layer(bn42)
#     conv43 = layer.conv_layer(relu42, name = "conv43", filt_shape = [3, 3, channels[4], channels[4]])
#     bn43 = layer.batch_normalization(conv43, name = "bn43")
#     relu43 = layer.relu_layer(bn43)
#     conv44 = layer.conv_layer(relu43, name = "conv44", filt_shape = [3, 3, channels[4], channels[4]])
#     output44 = conv44 + output41
#     bn44 = layer.batch_normalization(output44, "bn44")
#     relu44 = layer.relu_layer(bn44)

#     # avg pooling
#     avg_pool = layer.avg_pool_layer(relu44, name = "avg_pool", pooling_size = [1, 8, 8, 1])

#     # flatten and fully connected
#     flatten = layer.flatten_layer(avg_pool)
#     fc = layer.fc_layer(flatten, num_classes, "prediction")

#     return fc

def basic(x, name, in_channel, out_channel):
    conv0 = layer.bn_relu_conv(x, name + "_basic0", in_channel, out_channel, 3)
    conv1 = layer.bn_relu_conv(conv0, name + "_basic1", out_channel, out_channel, 3) 
    return conv1


def bottle_neck(x, name, in_channel, out_channel):
    conv0 = layer.bn_relu_conv(x, name + "_bottleneck0", in_channel, out_channel, 3)
    conv1 = layer.bn_relu_conv(conv0, name + "_bottleneck1", out_channel, out_channel / 2, 3)
    conv2 = layer.bn_relu_conv(conv1, name + "_bottleneck2", out_channel / 2, out_channel, 3)
    return conv2

def basic_wide(x, name, in_channel, out_channel):
    conv0 = layer.bn_relu_conv(x, name + "_basicwide0", in_channel, out_channel, 3)
    conv1 = layer.bn_relu_conv(conv0, name + "_basicwide1", out_channel, out_channel, 3) 
    return conv1

def dropout(x, name, prob, in_channel, out_channel):
    conv0 = layer.bn_relu_conv(x, name + "_dropout_conv0", in_channel, out_channel, 3)
    dropout1 = tf.nn.dropout(conv0, prob, name = name + "_dropout")
    conv2 = layer.bn_relu_conv(dropout1, name + "_dropout_conv1", out_channel, out_channel, 3)
    return conv2


def build_wide_resnet(x, num_classes, N, k, block, prob = None):
    channels = [3, 16, 16 * k, 32 * k, 64 * k]
    layers = []

    # conv1
    # conv1 = layer.bn_relu_conv(x, "conv1", channels[0], channels[1], 3)
    conv1 = layer.conv_bn_relu(x, "conv1", channels[0], channels[1], 3)
    layers.append(conv1)

    # conv2
    # 1st
    before20 = layers[-1]
    conv20 = layer.conv_layer(before20, "conv20", [3, 3, channels[1], channels[2]])
    # conv20b = block(before20, "conv20b", prob, channels[1], channels[2]) if block is dropout else block(before20, "conv20b", channels[1], channels[2])
    conv20b_ = layer.conv_bn_relu(before20, "conv20b_", channels[1], channels[2], 3)
    conv20b = layer.conv_layer(conv20b_, "conv20b", [3, 3, channels[2], channels[2]])
    output20 = layer.bn_relu(conv20 + conv20b, "output20")
    layers.append(output20)

    # others
    for n in range(1, N):
        before2n = tf.identity(layers[-1])
        # conv2n = layer.conv_layer(before2n, "conv2%d" % n, [3, 3, channels[2], channels[2]])
        conv2nb = block(layers[-1], "conv2%db" % n, prob, channels[2], channels[2]) if block is dropout else block(layers[-1], "conv2%db" % n, channels[2], channels[2])
        output2n = layer.bn_relu(before2n + conv2nb, "output2%d" % n)
        layers.append(output2n)

    # downsampling0
    #downsampling0 = layer.avg_pool_layer(layers[-1], "downsampling0", [1, 2, 2, 1])
    downsampling0 = layer.max_pool_layer(layers[-1], "downsampling0", [1, 2, 2, 1])
    layers.append(downsampling0)

    # conv3
    # 1st
    before30 = layers[-1]
    conv30 = layer.conv_layer(before30, "conv30", [3, 3, channels[2], channels[3]])
    # conv30b = block(before30, "conv30b", prob, channels[2], channels[3]) if block is dropout else block(before30, "conv30b", channels[2], channels[3])
    conv30b_ = layer.conv_bn_relu(before30, "conv30b_", channels[2], channels[3], 3)
    conv30b = layer.conv_layer(conv30b_, "conv30b", [3, 3, channels[3], channels[3]])
    output30 = layer.bn_relu(conv30 + conv30b, "output30")
    layers.append(output30)

    # others
    for n in range(1, N):
        before3n = tf.identity(layers[-1])
        # conv3n = layer.conv_layer(before3n, "conv3%d" % n, [3, 3, channels[3], channels[3]])
        conv3nb = block(layers[-1], "conv3%db" % n, prob, channels[3], channels[3]) if block is dropout else block(layers[-1], "conv3%db" % n, channels[3], channels[3])
        output3n = layer.bn_relu(before3n + conv3nb, "output3%d" % n)
        layers.append(output3n)

    # downsampling1
    #downsampling1 = layer.avg_pool_layer(layers[-1], "downsampling1", [1, 2, 2, 1])
    downsampling1 = layer.max_pool_layer(layers[-1], "downsampling1", [1, 2, 2, 1])
    layers.append(downsampling1)

    # conv4
    # 1st
    before40 = layers[-1]
    conv40 = layer.conv_layer(before40, "conv40", [3, 3, channels[3],channels[4]])
    # conv40b = block(before40, "conv40b", prob, channels[3], channels[4]) if block is dropout else block(before40, "conv40b", channels[3], channels[4])
    conv40b_ = layer.conv_bn_relu(before40, "conv40b_", channels[3], channels[4], 3)
    conv40b = layer.conv_layer(conv40b_, "conv40b", [3, 3, channels[4], channels[4]])
    output40 = layer.bn_relu(conv40 + conv40b, "output40")
    layers.append(output40)

    # others
    for n in range(1, N):
        before4n = tf.identity(layers[-1])
        # conv4n = layer.conv_layer(before4n, "conv4%d" % n, [3, 3, channels[4], channels[4]])
        conv4nb = block(layers[-1], "conv4%db" % n, prob, channels[4], channels[4]) if block is dropout else block(layers[-1], "conv4%db" % n, channels[4], channels[4])
        output4n = layer.bn_relu(before4n + conv4nb, "output4%d" % n)
        layers.append(output4n)

    # avg pooling
    avg_pool = layer.avg_pool_layer(layers[-1], name = "avg_pool", pooling_size = [1, 8, 8, 1])
    layers.append(avg_pool)

    # flatten and fully connected
    flatten = layer.flatten_layer(layers[-1])
    fc = layer.fc_layer(flatten, num_classes, "fc")
    layers.append(fc)
    
    sm = tf.nn.softmax(layers[-1], name = "prediction")
    layers.append(sm)

    return layers[-1]

def test_wide_resnet():
    y = np.random.random((128, 32, 32, 3))
    x = tf.constant(y, dtype = tf.float32)
    pred = build_wide_resnet(x, 10, 4, 10, basic)
    sess = tf.Session()
    print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    sess.run(tf.global_variables_initializer())
    print "OK!"
    print pred

# test_wide_resnet()