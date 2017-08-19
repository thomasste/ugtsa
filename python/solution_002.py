from algorithms.ucb_mcts.algorithm import Algorithm as UCBAlgorithm
from games.game.game_state import GameState
from computation_graphs.computation_graph.computation_graph \
    import ComputationGraph
from computation_graphs.shiftable_computation_graph.computation_graph \
    import ComputationGraph as ShiftableComputationGraph
from computation_graphs.synchronized_computation_graph.computation_graph \
    import ComputationGraph as SynchronizedComputationGraph
from config import config
from copy import deepcopy
from model_builders.model_builder.model_builder import ModelBuilder
from threading import Lock, Semaphore, Thread

import argparse
import logging
import numpy as np
import sys
import tensorflow as tf
import threading
import time


logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('game', type=str)
argument_parser.add_argument('ugtsa_algorithm', type=str)
argument_parser.add_argument('model_builder', type=str)
argument_parser.add_argument('thread_count', type=int)
argument_parser.add_argument('compute_gradient_each', type=int)
argument_parser.add_argument('compute_gradient_times', type=int)
argument_parser.add_argument('ucb_worker_count', type=int)
argument_parser.add_argument('ucb_grow_factor', type=int)
argument_parser.add_argument('ugtsa_worker_count', type=int)
argument_parser.add_argument('ugtsa_grow_factor', type=int)
argument_parser.add_argument('--debug', action='store_true')
argument_parser.add_argument('--use_initial_state', action='store_true')
args = argument_parser.parse_args()

game_config = config['games'][args.game]
game_state = game_config['game_state']
ugtsa_algorithm_config = game_config['algorithms'][args.ugtsa_algorithm]
ugtsa_algorithm_class = ugtsa_algorithm_config['class']
model_builder = ugtsa_algorithm_config['model_builders'][args.model_builder]


class ThreadSynchronizer(object):
    def __init__(self, thread_count):
        self.thread_count = thread_count
        self.lock = Lock()
        self.threads = {}

    def run_when_all_threads_join(self, runnable):
        semaphore = None

        with self.lock:
            if len(self.threads) == self.thread_count - 1:
                runnable()
                for thread in self.threads.values():
                    thread.release()
                self.threads = {}
            else:
                semaphore = Semaphore(value=0)
                self.threads[threading.get_ident()] = semaphore

        if semaphore is not None:
            semaphore.acquire()


class OracleThread(Thread):
    def __init__(self, ucb_algorithm, ugtsa_algorithm):
        super().__init__()
        self.ucb_algorithm = ucb_algorithm
        self.ugtsa_algorithm = ugtsa_algorithm

        self.lock = Lock()

        self.training_thread_is_waiting = False
        self.training_thread_semaphore = Semaphore(value=0)
        self.ucb_move_rates = None
        self.ugtsa_move_rates = None

        self.should_finish = False

    def run(self):
        logger.info('{}: oracle - started'.format(threading.get_ident()))
        self.ugtsa_algorithm.computation_graph.add_thread()

        improve_counter = 0

        while True:
            self.ucb_algorithm.improve()
            self.ugtsa_algorithm.improve()
            improve_counter += 1

            with self.lock:
                if self.training_thread_is_waiting:
                    logger.info('{}: oracle - ucb {} {}'.format(
                        threading.get_ident(),
                        improve_counter,
                        [x.number_of_visits
                         for x in self.ucb_algorithm.tree[:20]]))
                    logger.info('{}: oracle - ugtsa {} {}'.format(
                        threading.get_ident(),
                        improve_counter,
                        [x.number_of_visits
                         for x in self.ugtsa_algorithm.tree[:20]]))
                    self.ucb_move_rates = [
                        self.ucb_algorithm.value(move_rate)
                        for move_rate in self.ucb_algorithm.move_rates()]
                    self.ugtsa_move_rates = [
                        self.ugtsa_algorithm.value(move_rate)
                        for move_rate in self.ugtsa_algorithm.move_rates()]

                    self.training_thread_is_waiting = False
                    self.training_thread_semaphore.release()

                if self.should_finish:
                    break

        self.ugtsa_algorithm.computation_graph.remove_thread()
        logger.info('{}: oracle finished'.format(threading.get_ident()))


first_node = 0
move_rate_dict = {}


class TrainingThread(Thread):
    def __init__(
            self, ugtsa_algorithm, oracle_thread: OracleThread,
            thread_synchronizer: ThreadSynchronizer):
        super().__init__()
        self.ugtsa_algorithm = ugtsa_algorithm
        self.oracle_thread = oracle_thread

        self.thread_synchronizer = thread_synchronizer

    def __apply_gradients(self):
        global first_node
        global move_rate_dict

        logger.info('{}: training - apply gradients begin'.format(
            threading.get_ident()))

        # TODO: cost function
        # TODO: decrease ucb impact
        y_grads = {
            move_rate: ugtsa_move_rate - oracle_ucb_move_rate
            for move_rate, (
                ugtsa_move_rate,
                oracle_ucb_move_rate,
                oracle_ugtsa_move_rate) in move_rate_dict.items()}

        self.ugtsa_algorithm.computation_graph.model_gradients(
            first_node=first_node,
            y_grads=y_grads)

        first_node = len(
            self.ugtsa_algorithm.computation_graph.computation_graph.nodes)
        move_rate_dict = {}

        # TODO: apply gradients
        logger.info('{}: training - apply gradients end'.format(
            threading.get_ident()))

    def run(self):
        logger.info('{}: training started'.format(threading.get_ident()))
        self.ugtsa_algorithm.computation_graph.add_thread()

        counter = 0
        last_gradient_time = time.time()
        improve_counter = 0

        while True:
            self.ugtsa_algorithm.improve()
            improve_counter += 1

            if time.time() - last_gradient_time > args.compute_gradient_each:
                logger.info('{}: training - compute_gradients begin'.format(
                    threading.get_ident()))
                logger.info('{}: training ucb {} {}'.format(
                    threading.get_ident(),
                    improve_counter,
                    [x.number_of_visits
                     for x in self.ugtsa_algorithm.tree[:20]]))

                # collect training data
                ugtsa_move_rates = self.ugtsa_algorithm.move_rates()
                # remove from computation_graph
                self.ugtsa_algorithm.computation_graph.remove_thread()

                with self.oracle_thread.lock:
                    self.oracle_thread.training_thread_is_waiting = True

                self.oracle_thread.training_thread_semaphore.acquire()

                # TODO: use only unmasked moves
                for move_rate, oracle_ucb_move_rate, \
                    oracle_ugtsa_move_rate in zip(
                        ugtsa_move_rates, self.oracle_thread.ucb_move_rates,
                        self.oracle_thread.ugtsa_move_rates):
                    move_rate_dict[move_rate] = (
                        self.ugtsa_algorithm.value(move_rate),
                        oracle_ucb_move_rate,
                        oracle_ugtsa_move_rate)

                self.oracle_thread.ucb_move_rates = None
                self.oracle_thread.ugtsa_move_rates = None

                # apply gradients
                self.thread_synchronizer.run_when_all_threads_join(
                    self.__apply_gradients)
                last_gradient_time = time.time()

                logger.info('{}: training - compute_gradients end'.format(
                    threading.get_ident()))

                # exit
                counter += 1
                if counter == args.compute_gradient_times:
                    break

                # enter computation graph
                self.ugtsa_algorithm.computation_graph.add_thread()

        with self.oracle_thread.lock:
            self.oracle_thread.should_finish = True

        logger.info('{}: training finished'.format(threading.get_ident()))


class GameStateThread(Thread):
    def __init__(
            self, game_state: GameState,
            thread_synchronizer: ThreadSynchronizer,

            training_computation_graph: ComputationGraph,
            training_empty_statistic: ComputationGraph.Transformation,
            training_move_rate: ComputationGraph.Transformation,
            training_game_state_as_update: ComputationGraph.Transformation,
            training_updated_statistic: ComputationGraph.Transformation,
            training_updated_update: ComputationGraph.Transformation,

            oracle_computation_graph: ComputationGraph,
            oracle_empty_statistic: ComputationGraph.Transformation,
            oracle_move_rate: ComputationGraph.Transformation,
            oracle_game_state_as_update: ComputationGraph.Transformation,
            oracle_updated_statistic: ComputationGraph.Transformation,
            oracle_updated_update: ComputationGraph.Transformation):
        super().__init__()
        self.game_state = game_state
        self.thread_synchronizer = thread_synchronizer

        self.training_computation_graph = training_computation_graph
        self.training_empty_statistic = training_empty_statistic
        self.training_move_rate = training_move_rate
        self.training_game_state_as_update = training_game_state_as_update
        self.training_updated_statistic = training_updated_statistic
        self.training_updated_update = training_updated_update

        self.oracle_computation_graph = oracle_computation_graph
        self.oracle_empty_statistic = oracle_empty_statistic
        self.oracle_move_rate = oracle_move_rate
        self.oracle_game_state_as_update = oracle_game_state_as_update
        self.oracle_updated_statistic = oracle_updated_statistic
        self.oracle_updated_update = oracle_updated_update

    def run(self):
        if args.use_initial_state:
            game_state = deepcopy(self.game_state)
        else:
            game_state = GameState.random_game_state(self.game_state)

        if game_state.is_final():
            game_state.undo_move()

        print(game_state)

        # TODO: removed moves
        removed_root_moves = []

        ucb_algorithm = UCBAlgorithm(
            game_state=deepcopy(game_state),
            worker_count=args.ucb_worker_count,
            grow_factor=args.ucb_grow_factor,
            exploration_factor=np.sqrt(2),
            removed_root_moves=removed_root_moves)

        oracle_ugtsa_algorithm = ugtsa_algorithm_class(
            game_state=deepcopy(game_state),
            worker_count=args.ugtsa_worker_count,
            grow_factor=args.ugtsa_grow_factor,
            removed_root_moves=removed_root_moves,
            computation_graph=self.oracle_computation_graph,
            empty_statistic=self.oracle_empty_statistic,
            move_rate=self.oracle_move_rate,
            game_state_as_update=self.oracle_game_state_as_update,
            updated_statistic=self.oracle_updated_statistic,
            updated_update=self.oracle_updated_update)

        training_ugtsa_algorithm = ugtsa_algorithm_class(
            game_state=deepcopy(game_state),
            worker_count=args.ugtsa_worker_count,
            grow_factor=args.ugtsa_worker_count,
            removed_root_moves=[],
            computation_graph=self.training_computation_graph,
            empty_statistic=self.training_empty_statistic,
            move_rate=self.training_move_rate,
            game_state_as_update=self.training_game_state_as_update,
            updated_statistic=self.training_updated_statistic,
            updated_update=self.training_updated_update)

        oracle_thread = OracleThread(
            ucb_algorithm=ucb_algorithm,
            ugtsa_algorithm=oracle_ugtsa_algorithm)

        training_thread = TrainingThread(
            ugtsa_algorithm=training_ugtsa_algorithm,
            oracle_thread=oracle_thread,
            thread_synchronizer=self.thread_synchronizer)

        oracle_thread.start()
        training_thread.start()

        oracle_thread.join()
        training_thread.join()

graph = tf.Graph()
model_builder.worker_count = args.ugtsa_worker_count
with graph.as_default():
    model_builder.build()

config = tf.ConfigProto()
if args.debug:
    config.log_device_placement = True

with tf.Session(config=config, graph=graph) as session:
    session.run(tf.global_variables_initializer())

    oracle_computation_graph = SynchronizedComputationGraph(
        ShiftableComputationGraph(False, session))
    oracle_empty_statistic, oracle_move_rate, oracle_game_state_as_update, \
        oracle_updated_statistic, oracle_updated_update = \
        ModelBuilder.transformations(oracle_computation_graph, graph)

    training_computation_graph = SynchronizedComputationGraph(
        ShiftableComputationGraph(True, session))
    training_empty_statistic, training_move_rate, \
        training_game_state_as_update, training_updated_statistic, \
        training_updated_update = \
        ModelBuilder.transformations(training_computation_graph, graph)

    thread_synchronizer = ThreadSynchronizer(args.thread_count)

    game_state_threads = [
        GameStateThread(
            game_state=deepcopy(game_state),
            thread_synchronizer=thread_synchronizer,
            training_computation_graph=training_computation_graph,
            training_empty_statistic=training_empty_statistic,
            training_move_rate=training_move_rate,
            training_game_state_as_update=training_game_state_as_update,
            training_updated_statistic=training_updated_statistic,
            training_updated_update=training_updated_update,
            oracle_computation_graph=oracle_computation_graph,
            oracle_empty_statistic=oracle_empty_statistic,
            oracle_move_rate=oracle_move_rate,
            oracle_game_state_as_update=oracle_game_state_as_update,
            oracle_updated_statistic=oracle_updated_statistic,
            oracle_updated_update=oracle_updated_update)
        for _ in range(args.thread_count)]

    for thread in game_state_threads:
        thread.start()

    for thread in game_state_threads:
        thread.join()
