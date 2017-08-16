from config import config
from games.game.game_state import GameState
from copy import deepcopy
from algorithms.ucb_mcts.algorithm import Algorithm as UCBAlgorithm
from computation_graphs.shiftable_computation_graph.computation_graph import \
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
args = argument_parser.parse_args()

game_config = config['games'][args.game]

game_state = game_config['game_state']
algorithm_config = game_config['algorithms'][args.algorithm]
algorithm_class = algorithm_config['class']
model_builder = algorithm_config['model_builders'][args.model_builder]

graph = tf.Graph()
model_builder.worker_count = args.ugtsa_worker_count
with graph.as_default():
    model_builder.build()

config = tf.ConfigProto()
if args.debug == True:
    config.log_device_placement=True

with tf.Session(config=config, graph=graph) as session:
    session.run(tf.global_variables_initializer())

    computation_graph = ComputationGraph(True, session)
    empty_statistic, move_rate, game_state_as_update, \
        updated_statistic, updated_update = \
        ModelBuilder.transformations(computation_graph, graph)

    for i in range(args.number_of_iterations):
        first_node = computation_graph.nodes_shift + \
                     len(computation_graph.nodes)
        computation_graph.shift(first_node)

        gs = GameState.random_game_state(game_state)
        # gs = game_state

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
            empty_statistic=empty_statistic,
            move_rate=move_rate,
            game_state_as_update=game_state_as_update,
            updated_statistic=updated_statistic,
            updated_update=updated_update)
        for _ in range(args.ugtsa_strength):
            ugtsa_algorithm.improve()
        ugtsa_end = time.time()
        logger.info('{}: UGTSA took {}'.format(i, ugtsa_end - ugtsa_begin))

        # TODO: cost function
        gradients_begin = time.time()
        gradients = computation_graph.model_gradients(
            first_node=first_node,
            y_grads={
                ugtsa_move_rate:
                    ugtsa_algorithm.value(ugtsa_move_rate) -
                    ucb_algorithm.value(ucb_move_rate)
                for ugtsa_move_rate, ucb_move_rate in zip(
                    ugtsa_algorithm.move_rates(), ucb_algorithm.move_rates())})
        gradients_end = time.time()
        logger.info(
            '{}: gradients took {}'.format(i, gradients_end - gradients_begin))

        # debug
        print([x.number_of_visits for x in ucb_algorithm.tree[:20]])
        print([x.number_of_visits for x in ugtsa_algorithm.tree[:20]])

        if args.debug:
            print(ucb_algorithm.tree[:20])
            print(ugtsa_algorithm.tree[:20])
            print(gradients)
