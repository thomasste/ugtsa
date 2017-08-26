from model_builders.model_builder import model_builder
from model_builders.common import *

import tensorflow as tf


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(self, model_builder: model_builder.ModelBuilder,
                 updated_update_lstm_state_sizes):
        super().__init__(
            model_builder.player_count, model_builder.worker_count,
            model_builder.statistic_size, model_builder.update_size,
            model_builder.game_state_board_shape,
            model_builder.game_state_statistic_size,
            model_builder.update_statistic_size,
            model_builder.seed_size)
        self.model_builder = model_builder
        self.updated_update_lstm_state_sizes = updated_update_lstm_state_sizes
        assert self.update_size == 2 * sum(updated_update_lstm_state_sizes)

    def _empty_statistic(self, training, global_step, seed, game_state_board,
                         game_state_statistic):
        return self.model_builder._empty_statistic(
            training, global_step, seed, game_state_board,
            game_state_statistic)

    def _move_rate(self, training, global_step, seed, parent_statistic,
                   child_statistic):
        return self.model_builder._move_rate(
            training, global_step, seed, parent_statistic, child_statistic)

    def _game_state_as_update(self, training, global_step, seed,
                              update_statistic):
        return self.model_builder._game_state_as_update(
            training, global_step, seed, update_statistic)

    def _updated_statistic(self, training, global_step, seed, statistic,
                           update_count, updates):
        return self._updated_statistic(
            training, global_step, seed, statistic, update_count, updates)

    def _updated_update(self, training, global_step, seed, update, statistic):
        split = tf.split(
            update, self.updated_update_lstm_state_sizes * 2, 1)

        old_states = split[:len(self.updated_update_lstm_state_sizes)]
        old_outputs = split[len(self.updated_update_lstm_state_sizes):]

        new_states, new_outputs = [], []

        input = statistic
        for index, (old_state, old_output) in enumerate(
                zip(old_states, old_outputs)):
            with tf.variable_scope('lstm_layer_{}'.format(index)):
                print(old_state.get_shape(), old_output.get_shape())
                state, output = lstm(input, old_state, old_output)
                new_states += [state]
                new_outputs += [output]

                input = output

        return tf.concat(new_states + new_outputs, axis=1)

    def _cost_function(self, training, global_step, move_rate, ucb_move_rate,
                       ugtsa_move_rate, trainable_variables):
        return self.model_builder._cost_function(
            training, global_step, move_rate, ucb_move_rate, ugtsa_move_rate,
            trainable_variables)

    def _apply_gradients(self, training, global_step, grads_and_vars):
        return self.model_builder._apply_gradients(
            training, global_step, grads_and_vars)
