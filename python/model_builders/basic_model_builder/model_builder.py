from model_builders.model_builder import model_builder
from model_builders.common import *

import tensorflow as tf
import numpy as np


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(self, player_count, worker_count, statistic_size, update_size,
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
                 cost_function_ucb_half_life,
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
        self.cost_function_ucb_half_life = cost_function_ucb_half_life
        self.cost_function_regularization_factor = \
            cost_function_regularization_factor

    def _empty_statistic(self, training, global_step, seed, game_state_board,
                         game_state_statistic):
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
                    seed, signal, training=training)
                print(signal.get_shape())

        return signal

    def _move_rate(self, training, global_step, seed, parent_statistic,
                   child_statistic):
        signal = tf.concat([parent_statistic, child_statistic], axis=1)
        print(signal.get_shape())

        for idx, output_size in enumerate(
                self.move_rate_hidden_output_sizes):
            with tf.variable_scope('dense_layer_{}'.format(idx)):
                signal = dense_layer(signal, output_size)
                signal = bias_layer(signal)
                signal = activation_layer(signal)
                seed, signal = dropout_layer(
                    seed, signal, training=training)
                print(signal.get_shape())

        signal = dense_layer(signal, self.player_count)
        signal = bias_layer(signal)
        signal = tf.nn.softmax(signal)
        print(signal.get_shape())

        return signal

    def _game_state_as_update(self, training, global_step, seed,
                              update_statistic):
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
                    seed, signal, training=training)
                print(signal.get_shape())

        return signal

    def _updated_statistic(self, training, global_step, seed, statistic,
                           update_count, updates):

        loop_condition = lambda i, updates, states, outputs: tf.less(i, tf.reduce_max(update_count))
        def loop_body(i, updates, states, outputs):
            input = updates[:, i*self.update_size: (i+1)*self.update_size]
            new_states = []
            new_outputs = []
            for layer, (state, output) in enumerate(zip(states, outputs)):
                with tf.variable_scope('lstm_layer_{}'.format(layer), reuse=True):
                    modified_state, modified_output = lstm(input, state, output)
                    print(
                        input.get_shape(),
                        modified_state.get_shape(),
                        modified_output.get_shape())
                    new_states += [tf.where(update_count > i, modified_state, state)]
                    new_outputs += [tf.where(update_count > i, modified_output, output)]
                    input = output
            return i+1, updates, new_states, new_outputs

        initial_states = [
            tf.tile(tf.Variable(
                name='initial_state',
                initial_value=tf.zeros((1, state_size))),
                [tf.shape(updates)[0], 1])
            for state_size in self.updated_statistic_lstm_state_sizes]
        initial_outputs = [
            tf.tile(tf.Variable(
                name='initial_output',
                initial_value=tf.zeros((1, state_size))),
                [tf.shape(updates)[0], 1])
            for state_size in self.updated_statistic_lstm_state_sizes]

        input = updates[:, 0: self.update_size]
        new_states = []
        new_outputs = []
        for layer, (state, output) in enumerate(zip(initial_states, initial_outputs)):
            with tf.variable_scope('lstm_layer_{}'.format(layer), reuse=False):
                modified_state, modified_output = lstm(input, state, output)
                print(
                    input.get_shape(),
                    modified_state.get_shape(),
                    modified_output.get_shape())
                new_states += [tf.where(update_count > 0, modified_state, state)]
                new_outputs += [tf.where(update_count > 0, modified_output, output)]
                input = output

        output_i, output_updates, output_states, output_outputs = \
            tf.while_loop(loop_condition, loop_body, (tf.constant(1), updates, new_states, new_outputs))

        signal = output_outputs[-1]
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
                    seed, signal, training=training)
                print(signal.get_shape())

        return signal

    def _updated_update(self, training, global_step, seed, update, statistic):
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
                    seed, signal, training=training)
                print(signal.get_shape())

        return signal

    def _cost_function(self, training, global_step, move_rate, ucb_move_rate,
                       ugtsa_move_rate, trainable_variables):
        # labels
        global_step_as_float = tf.cast(global_step, tf.float32)
        alpha = 1 - (global_step_as_float /
                     (self.cost_function_ucb_half_life + global_step_as_float))

        ucb_move_rate /= tf.reshape(
            tf.reduce_sum(ucb_move_rate, axis=1), (-1, 1))

        labels = alpha * ucb_move_rate + (1 - alpha) * ugtsa_move_rate

        # output
        output_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=move_rate, labels=labels))

        # regularization
        regularization_loss = tf.reduce_sum([
            tf.nn.l2_loss(variable)
            for variable in trainable_variables
            if 'bias' not in variable.name])

        return output_loss + \
            self.cost_function_regularization_factor * regularization_loss

    def _apply_gradients(self, training, global_step, grads_and_vars):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.apply_gradients(
            grads_and_vars, global_step, 'apply_gradients')
