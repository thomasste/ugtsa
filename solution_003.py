from algorithms.ucb_mcts.algorithm import Algorithm as UCBAlgorithm
from computation_graphs.shiftable_computation_graph.computation_graph \
    import ComputationGraph
from copy import deepcopy
from config import config
from games.game.game_state import GameState
from model_builders.model_builder.model_builder import ModelBuilder
from threading import Lock, Semaphore, Thread

import argparse
import logging
import numpy as np
import sys
import time
import tensorflow as tf
import threading

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
    def __init__(self, game_state: GameState, session: tf.Session):
        super().__init__()

        # TODO: limit number of moves
        self.removed_root_moves = []

        self.ucb_algorithm = UCBAlgorithm(
            game_state=deepcopy(game_state),
            worker_count=args.ucb_worker_count,
            grow_factor=args.ucb_grow_factor,
            exploration_factor=np.sqrt(2),
            removed_root_moves=self.removed_root_moves)

        computation_graph = ComputationGraph(
            training=False,
            session=session)
        transformations = ModelBuilder.transformations(
            computation_graph, tf.get_default_graph())

        self.ugtsa_algorithm = ugtsa_algorithm_class(
            game_state=deepcopy(game_state),
            worker_count=args.ugtsa_worker_count,
            grow_factor=args.ugtsa_grow_factor,
            removed_root_moves=self.removed_root_moves,
            computation_graph=computation_graph,
            empty_statistic=transformations[0],
            move_rate=transformations[1],
            game_state_as_update=transformations[2],
            updated_statistic=transformations[3],
            updated_update=transformations[4])

        self.lock = Lock()
        self.should_finish = False

    def run(self):
        logger.info('{}: oracle - started'.format(threading.get_ident()))

        while True:
            with self.lock:
                if self.should_finish:
                    break

                self.ucb_algorithm.improve()
                self.ugtsa_algorithm.improve()

        logger.info('{}: oracle - finished'.format(threading.get_ident()))
        logger.info('{}: oracle - ucb {}'.format(
            threading.get_ident(),
            [x.number_of_visits
             for x in self.ucb_algorithm.tree[:20]]))
        logger.info('{}: oracle - ugtsa {}'.format(
            threading.get_ident(),
            [x.number_of_visits
             for x in self.ugtsa_algorithm.tree[:20]]))


class TrainingThread(Thread):
    def __init__(
            self, game_state: GameState, session: tf.Session,
            oracle_thread: OracleThread,
            thread_synchronizer: ThreadSynchronizer):
        super().__init__()

        self.computation_graph = ComputationGraph(
            training=False,
            session=session)
        transformations = ModelBuilder.transformations(
            self.computation_graph, tf.get_default_graph())

        self.algorithm = ugtsa_algorithm_class(
            game_state=deepcopy(game_state),
            worker_count=args.ugtsa_worker_count,
            grow_factor=args.ugtsa_grow_factor,
            removed_root_moves=[],
            computation_graph=self.computation_graph,
            empty_statistic=transformations[0],
            move_rate=transformations[1],
            game_state_as_update=transformations[2],
            updated_statistic=transformations[3],
            updated_update=transformations[4])

        self.oracle_thread = oracle_thread
        self.thread_synchronizer = thread_synchronizer

    def __apply_gradients(self):
        # TODO: apply_gradients
        logger.info('{}: training - apply gradients'.format(
            threading.get_ident()))

    def run(self):
        logger.info('{}: training - started'.format(threading.get_ident()))

        counter = 0
        last_gradient_time = time.time()

        while True:
            self.algorithm.improve()

            if time.time() - last_gradient_time > args.compute_gradient_each:
                logger.info('{}: training - compute gradients'.format(
                    threading.get_ident()))
                logger.info('{}: training - {}'.format(
                    threading.get_ident(),
                    [x.number_of_visits
                     for x in self.algorithm.tree[:20]]))

                # collect training data
                move_rates = self.algorithm.move_rates()

                with self.oracle_thread.lock:
                    removed_root_moves = \
                        self.oracle_thread.removed_root_moves
                    ucb_move_rates = [
                        self.oracle_thread.ucb_algorithm.value(move_rate)
                        for move_rate in
                        self.oracle_thread.ucb_algorithm.move_rates()]
                    ugtsa_move_rates = [
                        self.oracle_thread.ugtsa_algorithm.value(move_rate)
                        for move_rate in
                        self.oracle_thread.ugtsa_algorithm.move_rates()]

                # TODO: cost function
                y_grads = {
                    move_rate:
                        self.algorithm.value(move_rate) -
                        (ucb_move_rate + ugtsa_move_rate) / 2
                    for i, (move_rate, ucb_move_rate, ugtsa_move_rate) in
                        enumerate(
                            zip(move_rates, ucb_move_rates, ugtsa_move_rates))
                    if i not in removed_root_moves
                }

                # calculate gradients
                # TODO: first node
                self.computation_graph.model_gradients(0, y_grads)

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

        with self.oracle_thread.lock:
            self.oracle_thread.should_finish = True

        logger.info('{}: training - finished'.format(
            threading.get_ident()))


class GameStateThread(Thread):
    def __init__(
            self, game_state: GameState, session: tf.Session,
            thread_synchronizer: ThreadSynchronizer):
        super().__init__()

        self.game_state = GameState.random_game_state(game_state)
        if game_state.is_final():
            game_state.undo_move()

        self.oracle_thread = OracleThread(self.game_state, session)
        self.training_thread = TrainingThread(
            self.game_state, session, self.oracle_thread, thread_synchronizer)

    def run(self):
        self.oracle_thread.start()
        self.training_thread.start()

        self.oracle_thread.join()
        self.training_thread.join()


# build graph
graph = tf.Graph()
model_builder.worker_count = args.ugtsa_worker_count
with graph.as_default():
    model_builder.build()

config = tf.ConfigProto()
# if args.debug:
#     config.log_device_placement = True

with tf.Session(config=config, graph=graph) as session:
    session.run(tf.global_variables_initializer())

    thread_synchronizer = ThreadSynchronizer(args.thread_count)

    game_state_threads = [
        GameStateThread(
            game_state=deepcopy(game_state),
            session=session,
            thread_synchronizer=thread_synchronizer)
        for _ in range(args.thread_count)]

    for thread in game_state_threads:
        thread.start()

    for thread in game_state_threads:
        thread.join()
