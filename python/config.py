from games.omringa import game_state as omringa_game_state
from games.omringa import algorithm as omringa_algorithm
from model_builders.basic_model_builder import model_builder as \
    basic_model_builder
from model_builders.vertical_lstm_model_builder import model_builder as \
    vertical_lstm_model_builder

config = {
    'games': {
        'omringa': {
            'game_state': omringa_game_state.GameState(7),
            'algorithms': {
                'default': {
                    'class': omringa_algorithm.Algorithm,
                    'model_builders': {
                        'small_basic_model_builder':
                            basic_model_builder.ModelBuilder(
                                player_count=2,
                                worker_count=None,
                                statistic_size=40,
                                update_size=20,
                                game_state_board_shape=[7, 7],
                                game_state_statistic_size=2,
                                update_statistic_size=2,
                                seed_size=30,
                                empty_statistic_filter_shapes=[
                                    (2, 2, 16), (2, 2, 32)],
                                empty_statistic_window_shapes=[
                                    (1, 2, 2, 1), (1, 2, 2, 1)],
                                empty_statistic_hidden_output_sizes=[
                                    25, 25, 25],
                                move_rate_hidden_output_sizes=[
                                    25, 25, 25],
                                game_state_as_update_hidden_output_sizes=[
                                    25, 25, 25],
                                updated_statistic_lstm_state_sizes=[
                                    25, 25, 25],
                                updated_statistic_hidden_output_sizes=[
                                    25, 25, 25],
                                updated_update_hidden_output_sizes=[
                                    25, 25, 25]),
                        'small_vertical_lstm_model_builder':
                            vertical_lstm_model_builder.ModelBuilder(
                                player_count=2,
                                worker_count=None,
                                statistic_size=40,
                                update_size=150,
                                game_state_board_shape=[7, 7],
                                game_state_statistic_size=2,
                                update_statistic_size=2,
                                seed_size=30,
                                empty_statistic_filter_shapes=[
                                    (2, 2, 16), (2, 2, 32)],
                                empty_statistic_window_shapes=[
                                    (1, 2, 2, 1), (1, 2, 2, 1)],
                                empty_statistic_hidden_output_sizes=[
                                    25, 25, 25],
                                move_rate_hidden_output_sizes=[
                                    25, 25, 25],
                                game_state_as_update_hidden_output_sizes=[
                                    25, 25, 25],
                                updated_statistic_lstm_state_sizes=[
                                    25, 25, 25],
                                updated_statistic_hidden_output_sizes=[
                                    25, 25, 25],
                                updated_update_lstm_state_sizes=[
                                    25, 25, 25]),
                    },
                },
            },
        },
    },
}
