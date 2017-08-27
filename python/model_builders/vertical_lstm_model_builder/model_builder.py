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

    def set_player_count(self, player_count):
        self.player_count = player_count
        self.model_builder.set_player_count(player_count)

    def set_worker_count(self, worker_count):
        self.worker_count = worker_count
        self.model_builder.set_worker_count(worker_count)

    def set_statistic_size(self, statistic_size):
        self.statistic_size = statistic_size
        self.model_builder.set_statistic_size(statistic_size)

    def set_update_size(self, update_size):
        self.update_size = update_size
        self.model_builder.set_update_size(update_size)

    def set_game_state_board_shape(self, game_state_board_shape):
        self.game_state_board_shape = game_state_board_shape
        self.model_builder.set_game_state_board_shape(game_state_board_shape)

    def set_game_state_statistic_size(self, game_state_statistic_size):
        self.game_state_statistic_size = game_state_statistic_size
        self.model_builder.set_game_state_statistic_size(
            game_state_statistic_size)

    def set_update_statistic_size(self, update_statistic_size):
        self.update_statistic_size = update_statistic_size
        self.model_builder.set_update_statistic_size(update_statistic_size)

    def set_seed_size(self, seed_size):
        self.seed_size = seed_size
        self.model_builder.set_seed_size(seed_size)

    def _empty_statistic(self, training, seed, game_state_board,
                         game_state_statistic):
        return self.model_builder._empty_statistic(
            training, seed, game_state_board,
            game_state_statistic)

    def _move_rate(self, training, seed, parent_statistic,
                   child_statistic):
        return self.model_builder._move_rate(
            training, seed, parent_statistic, child_statistic)

    def _game_state_as_update(self, training, seed, update_statistic):
        return self.model_builder._game_state_as_update(
            training, seed, update_statistic)

    def _updated_statistic(self, training, seed, statistic, update_count,
                           updates):
        return self.model_builder._updated_statistic(
            training, seed, statistic, update_count, updates)

    def _updated_update(self, training, seed, update, statistic):
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

    def _cost_function(self, global_step, move_rate, ucb_move_rate,
                       ugtsa_move_rate, trainable_variables):
        return self.model_builder._cost_function(
            global_step, move_rate, ucb_move_rate, ugtsa_move_rate,
            trainable_variables)

    def _apply_gradients(self, global_step, grads_and_vars):
        return self.model_builder._apply_gradients(
            global_step, grads_and_vars)
