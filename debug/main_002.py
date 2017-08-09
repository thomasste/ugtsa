from algorithms.tensorflow_mcts.algorithm import Algorithm
from model_builders.basic_model_builder.model_builder import ModelBuilder
from games.omringa.game_state import GameState

import tensorflow as tf
import time

game_state = GameState(7)
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
    updated_statistic_lstm_state_sizes=[25, 25, 25],
    updated_statistic_hidden_output_sizes=[25, 25, 25],
    updated_update_hidden_output_sizes=[25, 25, 25],
    cost_function_regularization_factor=0.001)

mb.build()

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)

    empty_statistic_model = session.run(
        tf.get_collection('basic/empty_statistic/model_initializer')[0])
    move_rate_model = session.run(
        tf.get_collection('basic/move_rate/model_initializer')[0])
    game_state_as_update_model = session.run(
        tf.get_collection('basic/game_state_as_update/model_initializer')[0])
    updated_statistic_model = session.run(
        tf.get_collection('basic/updated_statistic/model_initializer')[0])
    updated_update_model = session.run(
        tf.get_collection('basic/updated_update/model_initializer')[0])

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
        game_state=game_state,
        worker_count=worker_count,
        grow_factor=5,
        session=session,
        variable_scope='basic',
        training=True,
        empty_statistic_model=empty_statistic_model,
        move_rate_model=move_rate_model,
        game_state_as_update_model=game_state_as_update_model,
        updated_statistic_model=updated_statistic_model,
        updated_update_model=updated_update_model)

    start = time.time()

    for i in range(1000):
        a.improve()

    middle = time.time()

    print(a.computation_graph.gradients([
        a.empty_statistic_model,
        a.move_rate_model,
        a.game_state_as_update_model,
        a.updated_statistic_model,
        a.updated_update_model],
        len(a.computation_graph.nodes) - 6))

    end = time.time()

    print("forward: {} backward: {}".format(middle - start, end - middle))

    print([n for n in a.tree[:20]])
    print([n.number_of_visits for n in a.tree[:20]])
