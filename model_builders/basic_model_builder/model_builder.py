from model_builders.model_builder import model_builder
from tensorflow.contrib.stateless import stateless_random_uniform

import numpy as np
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


def dropout_layer(seed, signal, rate=0.5, training=False):
    s, seed = seed[:2], seed[2:]
    rand = stateless_random_uniform(tf.shape(signal), s)
    mask = tf.to_float(rand > rate)
    return seed, tf.cond(training, lambda: signal * mask, lambda: signal)


def lstm(signal, old_state, old_output):
    state_size = old_state.get_shape()[1].value

    merged_inputs = tf.concat([signal, old_output], axis=1)

    with tf.variable_scope('f'):
        f = tf.sigmoid(
            bias_layer(
                dense_layer(merged_inputs, state_size), constant=1.))
    with tf.variable_scope('i'):
        i = tf.sigmoid(bias_layer(dense_layer(merged_inputs, state_size)))
    with tf.variable_scope('o'):
        o = tf.sigmoid(bias_layer(dense_layer(merged_inputs, state_size)))
    with tf.variable_scope('state'):
        state = f * old_state + i * tf.tanh(
            bias_layer(dense_layer(merged_inputs, state_size)))
    output = o * tf.tanh(state)

    return state, output


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(
            self, player_count, worker_count, statistic_size, update_size,
            game_state_board_shape, game_state_statistic_size,
            update_statistic_size, seed_size,
            empty_statistic_filter_shapes,
            empty_statistic_window_shapes,
            empty_statistic_hidden_output_sizes,
            move_rate_hidden_output_sizes,
            game_state_as_update_hidden_output_sizes,
            updated_statistic_lstm_state_sizes,
            updated_statistic_hidden_output_sizes,
            updated_update_hidden_output_sizes,
            cost_function_regularization_factor):
        super().__init__(
            player_count, worker_count, statistic_size, update_size,
            game_state_board_shape, game_state_statistic_size,
            update_statistic_size, seed_size)

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
        self.cost_function_regularization_factor = \
            cost_function_regularization_factor

    def _empty_statistic_transformation(
            self, seed, game_state_board, game_state_statistic):
        signal = tf.expand_dims(game_state_board, -1)
        print(signal.get_shape())

        for idx, (filter_shape, window_shape) in \
                enumerate(zip(self.empty_statistic_filter_shapes,
                              self.empty_statistic_window_shapes)):
            with tf.variable_scope('convolutional_layer_{}'.format(idx)):
                signal = convolutional_layer(signal, filter_shape)
                signal = batch_normalization_layer(signal)
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
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _move_rate_transformation(
            self, seed, parent_statistic, child_statistic):
        signal = tf.concat([parent_statistic, child_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.move_rate_hidden_output_sizes):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=self.training)
                print(signal.get_shape())

        signal = dense_layer(signal, self.player_count)
        signal = bias_layer(signal)
        signal = tf.nn.softmax(signal)
        print(signal.get_shape())

        return signal

    def _game_state_as_update_transformation(self, seed, update_statistic):
        signal = update_statistic
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.game_state_as_update_hidden_output_sizes +
                [self.update_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _updated_statistic_transformation(
            self, seed, statistic, update_count, updates):
        inputs = [
            updates[:, i*self.update_size: (i+1)*self.update_size]
            for i in range(self.worker_count)]

        for index, state_size in enumerate(
                self.updated_statistic_lstm_state_sizes):
            with tf.variable_scope('lstm_layer_{}'.format(index)):
                states = [tf.tile(tf.Variable(
                    name='empty_state',
                    initial_value=tf.zeros((1, state_size))),
                    [tf.shape(updates)[0], 1])]
                outputs = [tf.tile(tf.Variable(
                    name='initial_state',
                    initial_value=tf.zeros((1, state_size))),
                    [tf.shape(updates)[0], 1])]

                for i in range(self.worker_count):
                    with tf.variable_scope('lstm', reuse=(i > 0)):
                        modified_state, modified_output = lstm(
                            inputs[i], states[-1], outputs[-1])
                        print(
                            inputs[i].get_shape(),
                            modified_output.get_shape(),
                            modified_output.get_shape())
                        states += [tf.where(
                            update_count > i, modified_state, states[-1])]
                        outputs += [tf.where(
                            update_count > i, modified_output, outputs[-1])]

                inputs = outputs[1:]

        signal = inputs[-1]
        print(signal.get_shape())
        signal = tf.concat([signal, statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.updated_statistic_hidden_output_sizes +
                [self.statistic_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=self.training)
                print(signal.get_shape())

        return signal

    def _updated_update_transformation(self, seed, update, statistic):
        signal = tf.concat([update, statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.updated_update_hidden_output_sizes +
                [self.update_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=self.training)
                print(signal.get_shape())

        return signal
