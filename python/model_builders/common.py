from tensorflow.contrib.stateless import stateless_random_uniform

import tensorflow as tf


def convolutional_layer(signal, filter_shape):
    filter_shape = [filter_shape[0], filter_shape[1],
                    signal.get_shape()[3].value, filter_shape[2]]
    filter = tf.get_variable(
        'filter', filter_shape, tf.float32,
        tf.contrib.layers.xavier_initializer())
    signal = tf.nn.conv2d(
        input=signal, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    return signal


def batch_normalization_layer(signal):
    batch_mean, batch_variance = tf.nn.moments(
        signal, list(range(signal.get_shape().ndims - 1)))

    gamma = tf.get_variable(
        'gamma', batch_mean.get_shape(), tf.float32, tf.ones_initializer())
    beta = tf.get_variable(
        'beta', batch_mean.get_shape(), tf.float32, tf.zeros_initializer())
    signal = signal - batch_mean
    signal /= tf.sqrt(batch_variance + 0.0001)
    signal = gamma * signal + beta
    return signal


def activation_layer(signal):
    return tf.nn.relu(signal)


def max_pool_layer(signal, window_shape):
    return tf.nn.max_pool(
        signal, ksize=window_shape, strides=window_shape, padding='SAME')


def dense_layer(signal, output_size):
    layer_shape = (signal.get_shape()[1].value, output_size)
    layer = tf.get_variable(
        'layer', layer_shape, tf.float32,
        tf.contrib.layers.xavier_initializer())
    signal = tf.matmul(signal, layer)
    return signal


def bias_layer(signal, constant=0.):
    bias = tf.get_variable(
        'bias', signal.get_shape()[1:], tf.float32,
        tf.constant_initializer(constant))
    return signal + bias


def dropout_layer(seed, signal, keep_prob=0.5, training=False):
    s, seed = seed[:2], seed[2:]
    rand = stateless_random_uniform(tf.shape(signal), s)
    mask = tf.to_float(rand < keep_prob)
    return seed, tf.cond(
        training,
        lambda: (signal * mask) / keep_prob,
        lambda: signal)
