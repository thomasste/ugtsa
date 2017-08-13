from model_builders.basic_model_builder import model_builder
from model_builders.common import *

import tensorflow as tf


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
            updated_update_lstm_state_sizes,
            cost_function_regularization_factor):
        super().__init__(
            player_count, worker_count, statistic_size, update_size,
            game_state_board_shape, game_state_statistic_size,
            update_statistic_size, seed_size,
            empty_statistic_filter_shapes,
            empty_statistic_window_shapes,
            empty_statistic_hidden_output_sizes,
            move_rate_hidden_output_sizes,
            game_state_as_update_hidden_output_sizes,
            updated_statistic_lstm_state_sizes,
            updated_statistic_hidden_output_sizes,
            [],
            cost_function_regularization_factor)
        self.updated_update_lstm_state_sizes = updated_update_lstm_state_sizes
        assert self.update_size == 2 * sum(updated_update_lstm_state_sizes)

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
