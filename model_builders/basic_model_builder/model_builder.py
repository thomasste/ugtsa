from model_builders.model_builder import model_builder
from tensorflow.contrib.stateless import stateless_random_uniform

import math
import numpy as np
import tensorflow as tf


def xavier_initializer(shape, input_size, output_size):
    stddev = math.sqrt(2 / input_size + output_size)
    return tf.truncated_normal(shape, stddev=stddev)


def convolutional_layer(model, signal, filter_shape):
    filter_shape = [filter_shape[0], filter_shape[1],
                    signal.get_shape()[3].value, filter_shape[2]]
    filter = model.get_variable(
        xavier_initializer(
            filter_shape,
            filter_shape[0] * filter_shape[1] * filter_shape[2],
            filter_shape[0] * filter_shape[1] * filter_shape[3]),
        name='filter')
    signal = tf.nn.conv2d(
        input=signal, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    return signal


def batch_normalization_layer(model, signal):
    batch_mean, batch_variance = tf.nn.moments(
        signal, list(range(signal.get_shape().ndims - 1)))

    gamma = model.get_variable(tf.ones(batch_mean.get_shape()), name='gamma')
    beta = model.get_variable(tf.zeros(batch_mean.get_shape()), name='beta')
    signal = signal - batch_mean
    signal /= tf.sqrt(batch_variance + 0.0001)
    signal = gamma * signal + beta
    return signal


def activation_layer(signal):
    return tf.sigmoid(signal)


def max_pool_layer(signal, window_shape):
    return tf.nn.max_pool(
        signal, ksize=window_shape, strides=window_shape, padding='SAME')


def dense_layer(model, signal, output_size, name='layer', reuse=False):
    layer_shape = (signal.get_shape()[1].value, output_size)
    layer = model.get_variable(
        xavier_initializer(layer_shape, layer_shape[0], layer_shape[1]),
        name=name, reuse=reuse)
    signal = tf.matmul(signal, layer)
    return signal


def bias_layer(model, signal, constant=0, name='bias', reuse=False):
    bias = model.get_variable(
        tf.constant(constant, tf.float32,
                    shape=(signal.get_shape()[1].value,)),
        name=name, reuse=reuse)
    return signal + bias


def dropout_layer(model, signal, rate=0.5, training=False):
    seed = tf.concat([model.get_seed(), model.get_seed()], axis=0)
    rand = stateless_random_uniform(tf.shape(signal), seed)
    mask = tf.to_float(rand > rate)
    return tf.cond(training, lambda: signal * mask, lambda: signal)


def lstm(model, signal, old_state, old_output):
    state_size = old_state.get_shape()[1].value

    f = tf.sigmoid(
        bias_layer(
            model,
            dense_layer(model, signal, state_size, name='W_f', reuse=True) +
            dense_layer(model, old_output, state_size, name='U_f',
                        reuse=True),
            name='B_f', reuse=True))
    i = tf.sigmoid(
        bias_layer(
            model,
            dense_layer(model, signal, state_size, name='W_i', reuse=True) +
            dense_layer(model, old_output, state_size, name='U_i',
                        reuse=True),
            name='B_i', reuse=True))
    o = tf.sigmoid(
        bias_layer(
            model,
            dense_layer(model, signal, state_size, name='W_o', reuse=True) +
            dense_layer(model, old_output, state_size, name='U_o',
                        reuse=True),
            name='B_o', reuse=True))
    state = f * old_state + i * tf.tanh(
        bias_layer(
            model,
            dense_layer(model, signal, state_size, name='W_state',
                        reuse=True) +
            dense_layer(model, old_output, state_size, name='U_state',
                        reuse=True),
            name='B_state', reuse=True))
    output = o * tf.tanh(state)

    return state, output


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(self, variable_scope, player_count, worker_count,
                 statistic_size, update_size, game_state_board_shape,
                 game_state_statistic_size, update_statistic_size,
                 empty_statistic_filter_shapes,
                 empty_statistic_window_shapes,
                 empty_statistic_hidden_output_sizes,
                 move_rate_hidden_output_sizes,
                 game_state_as_update_hidden_output_sizes,
                 updated_statistic_lstm_state_sizes,
                 updated_statistic_hidden_output_sizes,
                 updated_update_hidden_output_sizes):
        super().__init__(
            variable_scope, player_count, worker_count,
            statistic_size, update_size, game_state_board_shape,
            game_state_statistic_size, update_statistic_size)

        self.empty_statistic_filter_shapes = empty_statistic_filter_shapes
        self.empty_statistic_window_shapes = empty_statistic_window_shapes
        self.empty_statistic_hidden_output_sizes = \
            empty_statistic_hidden_output_sizes
        self.move_rate_hidden_output_sizes = \
            move_rate_hidden_output_sizes
        self.game_state_as_update_hidden_output_sizes = \
            game_state_as_update_hidden_output_sizes
        self.updated_statistic_lstm_state_sizes = \
            updated_statistic_lstm_state_sizes
        self.updated_statistic_hidden_output_sizes = \
            updated_statistic_hidden_output_sizes
        self.updated_update_hidden_output_sizes = \
            updated_update_hidden_output_sizes

    def _empty_statistic_transformation(
            self, model, game_state_board, game_state_statistic):
        signal = tf.expand_dims(game_state_board, -1)
        print(signal.get_shape())

        for idx, (filter_shape, window_shape) in \
                enumerate(zip(self.empty_statistic_filter_shapes,
                              self.empty_statistic_window_shapes)):
            with tf.variable_scope('convolutional_layer_{}'.format(idx)):
                signal = convolutional_layer(model, signal, filter_shape)
                signal = batch_normalization_layer(model, signal)
                signal = activation_layer(signal)
                print(signal.get_shape())
                signal = max_pool_layer(signal, window_shape)
                print(signal.get_shape())

        signal = tf.reshape(
            signal, (-1, np.prod(signal.get_shape().as_list()[1:])))
        print(signal.get_shape())

        signal = tf.concat([signal, game_state_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.empty_statistic_hidden_output_sizes +
                [self.statistic_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = bias_layer(model, signal)
                signal = activation_layer(signal)
                signal = dropout_layer(model, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _move_rate_transformation(
            self, model, parent_statistic, child_statistic):
        signal = tf.concat([parent_statistic, child_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.move_rate_hidden_output_sizes):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = bias_layer(model, signal)
                signal = activation_layer(signal)
                signal = dropout_layer(model, signal, training=self.training)
                print(signal.get_shape())

        signal = dense_layer(model, signal, self.player_count)
        signal = bias_layer(model, signal)
        signal = tf.nn.softmax(signal)
        print(signal.get_shape())

        return signal

    def _game_state_as_update_transformation(self, model, update_statistic):
        signal = update_statistic
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.game_state_as_update_hidden_output_sizes +
                [self.update_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = bias_layer(model, signal)
                signal = activation_layer(signal)
                signal = dropout_layer(model, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _updated_statistic_transformation(
            self, model, statistic, update_count, updates):
        inputs = [
            updates[:, i*self.update_size: (i+1)*self.update_size]
            for i in range(self.worker_count)]

        for idx, state_size in enumerate(
                self.updated_statistic_lstm_state_sizes):
            with tf.variable_scope('lstm_layer_{}'.format(idx)):
                states = [tf.tile(model.get_variable(
                           tf.zeros((1, state_size)),
                           name='empty_state'), [tf.shape(updates)[0], 1])]
                outputs = [tf.tile(model.get_variable(
                           tf.zeros((1, state_size)),
                           name='empty_input'), [tf.shape(updates)[0], 1])]

                for i in range(self.worker_count):
                    with tf.variable_scope('lstm'):
                        modified_state, modified_output = lstm(
                            model, inputs[i], states[-1], outputs[-1])
                        states += [tf.where(update_count > i,
                                            modified_state, states[-1])]
                        outputs += [tf.where(update_count > i,
                                             modified_output, outputs[-1])]

                inputs = outputs[1:]

        signal = inputs[-1]
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.updated_statistic_hidden_output_sizes +
                [self.statistic_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = bias_layer(model, signal)
                signal = activation_layer(signal)
                signal = dropout_layer(model, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _updated_update_transformation(self, model, update, statistic):
        signal = tf.concat([update, statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.updated_update_hidden_output_sizes +
                [self.update_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = bias_layer(model, signal)
                signal = activation_layer(signal)
                signal = dropout_layer(model, signal, training=self.training)
                print(signal.get_shape())

        return signal
