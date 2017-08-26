from config import config
from games.game.game_state import GameState
from copy import deepcopy
from algorithms.ucb_mcts.algorithm import Algorithm as UCBAlgorithm
from computation_graphs.basic_computation_graph.computation_graph import \
    ComputationGraph
from model_builders.model_builder.model_builder import ModelBuilder

import argparse
import logging
import numpy as np
import sys
import tensorflow as tf
import time

logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('game', type=str)
argument_parser.add_argument('algorithm', type=str)
argument_parser.add_argument('model_builder', type=str)
argument_parser.add_argument('number_of_iterations', type=int)
argument_parser.add_argument('ucb_worker_count', type=int)
argument_parser.add_argument('ucb_strength', type=int)
argument_parser.add_argument('ugtsa_worker_count', type=int)
argument_parser.add_argument('ugtsa_strength', type=int)
argument_parser.add_argument('--debug', action='store_true')
argument_parser.add_argument('--use_initial_state', action='store_true')
args = argument_parser.parse_args()

game_config = config['games'][args.game]

game_state = game_config['game_state']
algorithm_config = game_config['algorithms'][args.algorithm]
algorithm_class = algorithm_config['class']
model_builder = algorithm_config['model_builders'][args.model_builder]

graph = tf.Graph()
model_builder.set_worker_count(args.ugtsa_worker_count)
with graph.as_default():
    model_builder.build()

config = tf.ConfigProto()
config.log_device_placement = args.debug

with tf.Session(config=config, graph=graph) as session:
    session.run(tf.global_variables_initializer())
    for i in range(args.number_of_iterations):
        logger.info('{}: global_step {}'.format(i, session.run(
            graph.get_tensor_by_name('global_step:0'))))

        if args.use_initial_state:
            gs = GameState.random_game_state(game_state)
            if gs.is_final:
                gs.undo_move()
        else:
            gs = deepcopy(game_state)

        computation_graph = ComputationGraph(True)
        transformations = ModelBuilder.create_transformations(
            computation_graph)

        ucb_begin = time.time()
        ucb_algorithm = UCBAlgorithm(
            game_state=deepcopy(gs),
            worker_count=args.ucb_worker_count,
            grow_factor=5,
            exploration_factor=np.sqrt(2),
            removed_root_moves=[])
        for _ in range(args.ucb_strength):
            ucb_algorithm.improve()
        ucb_end = time.time()
        logger.info('{}: UCB took {}'.format(i, ucb_end - ucb_begin))

        ugtsa_begin = time.time()
        ugtsa_algorithm = algorithm_class(
            game_state=deepcopy(gs),
            worker_count=args.ugtsa_worker_count,
            grow_factor=5,
            removed_root_moves=[],
            computation_graph=computation_graph,
            empty_statistic=transformations[0],
            move_rate=transformations[1],
            game_state_as_update=transformations[2],
            updated_statistic=transformations[3],
            updated_update=transformations[4])
        for _ in range(args.ugtsa_strength):
            ugtsa_algorithm.improve()
        ugtsa_end = time.time()
        logger.info('{}: UGTSA took {}'.format(i, ugtsa_end - ugtsa_begin))

        gradients_begin = time.time()
        # zero gradient accumulators
        ModelBuilder.zero_model_gradient_accumulators()
        if args.debug:
            ModelBuilder.model_gradient_accumulators_debug_info()

        # calculate move rate gradients
        move_rates = ugtsa_algorithm.move_rates()

        move_rate_values = []
        ucb_move_rate_values = []
        ugtsa_move_rate_values = []
        for move_rate, ucb_move_rate in zip(
                move_rates, ucb_algorithm.move_rates()):
            move_rate_values += [ugtsa_algorithm.value(move_rate)]
            ucb_move_rate_values += [ucb_algorithm.value(ucb_move_rate)]
            ugtsa_move_rate_values += [
                np.zeros(gs.player_count, dtype=np.float32)]

        loss, move_rates_gradient = ModelBuilder.cost_function(
            move_rate_values, ucb_move_rate_values, ugtsa_move_rate_values)

        logger.info('{}: loss {}'.format(i, loss))
        if args.debug:
            print(move_rates_gradient)

        # accumulate gradients
        computation_graph.model_gradients(
            first_node=0,
            y_grads={
                move_rate: gradient
                for move_rate, gradient in zip(
                    move_rates, move_rates_gradient)})

        # apply gradients
        ModelBuilder.apply_gradients()
        if args.debug:
            ModelBuilder.model_gradient_accumulators_debug_info()
        gradients_end = time.time()
        logger.info('{}: gradients took {}'.format(
            i, gradients_end - gradients_begin))

        # debug
        print([x.number_of_visits for x in ucb_algorithm.tree[:20]])
        print([x.number_of_visits for x in ugtsa_algorithm.tree[:20]])

        if args.debug:
            print(ucb_algorithm.tree[:20])
            print(ugtsa_algorithm.tree[:20])
