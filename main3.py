from algorithms.tensorflow_mcts.algorithm import Algorithm
from model_builders.basic_model_builder.model_builder import ModelBuilder
from games.omringa.game_state import GameState
# from tensorflow.contrib.stateless import stateless_random_uniform

import tensorflow as tf

gs = GameState(7)
worker_count = 10

mb = ModelBuilder(
    variable_scope='basic',
    player_count=2,
    worker_count=worker_count,
    statistic_size=40,
    update_size=20,
    game_state_board_shape=[7, 7],
    game_state_statistic_size=2,
    update_statistic_size=2,
    empty_statistic_filter_shapes=[(2, 2, 16), (2, 2, 32)],
    empty_statistic_window_shapes=[(1, 2, 2, 1), (1, 2, 2, 1)],
    empty_statistic_hidden_output_sizes=[25, 25, 25],
    move_rate_hidden_output_sizes=[25, 25, 25],
    game_state_as_update_hidden_output_sizes=[25, 25, 25],
    updated_statistic_lstm_state_sizes=[50, 50, 50],
    updated_statistic_hidden_output_sizes=[25, 25, 25],
    updated_update_hidden_output_sizes=[25, 25, 25])

mb.build()

exit(0)

with tf.Session() as session:
    empty_statistic_model = session.run(
        tf.get_collection('basic/empty_statistic/initial_model')[0])
    move_rate_model = session.run(
        tf.get_collection('basic/move_rate/initial_model')[0])
    game_state_as_update_model = session.run(
        tf.get_collection('basic/game_state_as_update/initial_model')[0])
    updated_statistic_model = session.run(
        tf.get_collection('basic/updated_statistic/initial_model')[0])
    updated_update_model = session.run(
        tf.get_collection('basic/updated_update/initial_model')[0])

    print(empty_statistic_model)
    print(move_rate_model)
    print(game_state_as_update_model)
    print(updated_statistic_model)
    print(updated_update_model)

    print(empty_statistic_model.shape)
    print(move_rate_model.shape)
    print(game_state_as_update_model.shape)
    print(updated_statistic_model.shape)
    print(updated_update_model.shape)

    a = Algorithm(
        gs, 10, 5, 'basic', session,
        empty_statistic_model,
        move_rate_model,
        game_state_as_update_model,
        updated_statistic_model,
        updated_update_model)

    for i in range(10000):
        a.improve()

    print(a.tree[:20])
