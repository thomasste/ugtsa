from model_builders.model_builder import model_builder

import tensorflow as tf
import math


def convolutional_layer(model, signal, filter_shape):
    filter_shape = [filter_shape[0], filter_shape[1],
                    signal.get_shape()[3].value, filter_shape[2]]
    stddev = math.sqrt(
        2. / (filter_shape[0] * filter_shape[1] *
              (filter_shape[1] + filter_shape[2])))
    filter = model.get_variable(
        tf.truncated_normal(filter_shape, stddev=stddev),
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
    return tf.nn.relu(signal)


def max_pool_layer(signal, window_shape):
    return tf.nn.max_pool(
        signal, ksize=window_shape, strides=window_shape, padding='SAME')


def dense_layer(model, signal, output_size):
    layer_shape = (signal.get_shape()[1].value, output_size)
    stddev = math.sqrt(2. / (layer_shape[0] + layer_shape[1]))
    layer = model.get_variable(
        tf.truncated_normal(layer_shape, stddev=stddev), name='layer')
    signal = tf.matmul(signal, layer)
    return signal


def dropout_layer(signal, training):
    return signal
    # return tf.layers.dropout(signal, training=training)


def lstm(model, x, c, h):
    input_n = x.get_shape()[1].value
    hidden_n = h.get_shape()[1].value

    def weights(suffix):
        return model.get_variable(
            tf.random_normal(shape=(input_n + hidden_n, hidden_n)),
            name='W_{}'.format(suffix), reuse=True)

    def bias(suffix, value):
        return model.get_variable(
            tf.constant(value, shape=(hidden_n,)),
            name='B_{}'.format(suffix), reuse=True)

    signal = tf.concat([x, h], axis=1)
    print(x.get_shape(), c.get_shape(), h.get_shape(), signal.get_shape())
    i = tf.sigmoid(tf.matmul(signal, weights("i")) + bias("i", 0.))
    f = tf.sigmoid(tf.matmul(signal, weights("f")) + bias("f", 1.))
    o = tf.sigmoid(tf.matmul(signal, weights("o")) + bias("o", 0.))
    g = tf.tanh(tf.matmul(signal, weights("g")) + bias("g", 0.))

    modified_c = f * c + i * g

    return modified_c, o * tf.tanh(modified_c)


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(self, variable_scope, statistic_size, update_size,
                 game_state_board_shape, game_state_statistic_size,
                 update_statistic_size, player_count, worker_count,
                 empty_statistic_filter_shapes,
                 empty_statistic_window_shapes,
                 empty_statistic_hidden_output_sizes,
                 move_rate_hidden_output_sizes,
                 game_state_as_update_hidden_output_sizes,
                 updated_statistic_lstm_layers,
                 updated_statistic_hidden_output_sizes,
                 updated_update_hidden_output_sizes):
        super().__init__(
            variable_scope, statistic_size, update_size,
            game_state_board_shape, game_state_statistic_size,
            update_statistic_size, player_count, worker_count)

        self.empty_statistic_filter_shapes = empty_statistic_filter_shapes
        self.empty_statistic_window_shapes = empty_statistic_window_shapes
        self.empty_statistic_hidden_output_sizes = empty_statistic_hidden_output_sizes
        self.move_rate_hidden_output_sizes = move_rate_hidden_output_sizes
        self.game_state_as_update_hidden_output_sizes = game_state_as_update_hidden_output_sizes
        self.updated_statistic_lstm_layers = updated_statistic_lstm_layers
        self.updated_statistic_hidden_output_sizes = updated_statistic_hidden_output_sizes
        self.updated_update_hidden_output_sizes = updated_update_hidden_output_sizes

    def _empty_statistic_transformation(
            self, model, game_state_board, game_state_statistic):
        signal = tf.reshape(
            game_state_board,
            [-1, self.game_state_board_shape[0],
             self.game_state_board_shape[1], 1])
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

        # print([-1, signal.get_shape()[1].value * signal.get_shape()[2].value
        #        * signal.get_shape()[3].value])

        signal = tf.reshape(
            signal,
            [-1, signal.get_shape()[1].value * signal.get_shape()[2].value
             * signal.get_shape()[3].value])
        print(signal.get_shape())

        signal = tf.concat([signal, game_state_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.empty_statistic_hidden_output_sizes +
                [self.statistic_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = activation_layer(signal)
                signal = dropout_layer(signal, self.training)
                print(signal.get_shape())

        return signal

    def _move_rate_transformation(
            self, model, parent_statistic, child_statistic):
        signal = tf.concat([parent_statistic, child_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.move_rate_hidden_output_sizes + [self.player_count]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = activation_layer(signal)
                signal = dropout_layer(signal, self.training)
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
                signal = activation_layer(signal)
                signal = dropout_layer(signal, self.training)
                print(signal.get_shape())

        return signal

    def _updated_statistic_transformation(
            self, model, statistic, update_count, updates):
        input = [updates[:, i*self.update_size: (i+1)*self.update_size]
                 for i in range(self.worker_count)]

        for idx, hidden_size in enumerate(self.updated_statistic_lstm_layers):
            with tf.variable_scope('lstm_layer_{}'.format(idx)):
                c = [tf.tile(model.get_variable(
                        tf.zeros((1, hidden_size)),
                        name='empty_state'), [tf.shape(updates)[0], 1])]
                h = [tf.tile(model.get_variable(
                        tf.zeros((1, hidden_size)),
                        name='empty_input'), [tf.shape(updates)[0], 1])]

                for i in range(self.worker_count):
                    with tf.variable_scope('lstm'):
                        modified_c, modified_h = lstm(model, input[i], c[-1], h[-1])
                        c += [tf.where(update_count > i * self.update_size, modified_c, c[-1])]
                        h += [tf.where(update_count > i * self.update_size, modified_h, h[-1])]

                input = h[1:]

        signal = tf.concat([c[-1], h[-1]], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.updated_statistic_hidden_output_sizes +
                [self.statistic_size]):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(model, signal, output_size)
                signal = activation_layer(signal)
                signal = dropout_layer(signal, self.training)
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
                signal = activation_layer(signal)
                signal = dropout_layer(signal, self.training)
                print(signal.get_shape())

        return signal
