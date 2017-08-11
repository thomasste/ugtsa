from model_builders.model_builder import model_builder
from model_builders.common import *

import tensorflow as tf


class ModelBuilder(model_builder.ModelBuilder):
    def __init__(
            self, model_builder: model_builder.ModelBuilder,
            updated_update_lstm_state_sizes):
        super().__init__(
            model_builder.player_count, model_builder.worker_count,
            model_builder.statistic_size, model_builder.update_size,
            model_builder.game_state_board_shape,
            model_builder.game_state_statistic_size,
            model_builder.update_statistic_size, model_builder.seed_size)
        self.model_builder = model_builder
        self.updated_update_lstm_state_sizes = updated_update_lstm_state_sizes
        assert self.update_size == 2 * sum(updated_update_lstm_state_sizes)

    def _empty_statistic_transformation(
            self, seed, game_state_board, game_state_statistic):
        return self.model_builder._empty_statistic_transformation(
            seed, game_state_board, game_state_statistic)

    def _move_rate_transformation(
            self, seed, parent_statistic, child_statistic):
        return self.model_builder._move_rate_transformation(
            seed, parent_statistic, child_statistic)

    def _game_state_as_update_transformation(
            self, seed, update_statistic):
        return self.model_builder._game_state_as_update_transformation(
            seed, update_statistic)

    def _updated_statistic_transformation(
            self, seed, statistic, update_count, updates):
        return self.model_builder._updated_statistic_transformation(
            seed, statistic, update_count, updates)

    def _updated_update_transformation(self, seed, update, statistic):
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

    def build(self):
        with tf.variable_scope('settings'):
            self.training = tf.placeholder(tf.bool, name='training')
            self.model_builder.training = self.training

        self.__build_empty_statistic_graph()
        self.__build_move_rate_graph()
        self.__build_game_state_as_update_graph()
        self.__build_updated_statistic_graph()
        self.__build_updated_update_graph()
