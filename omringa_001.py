from algorithms.computation_graph_mcts.algorithm import Algorithm as UgtsaAlgorithm
from algorithms.ucb_mcts.algorithm import Algorithm as UcbAlgorithm
from copy import deepcopy
from games.omringa.game_state import GameState
from model_builders.basic_model_builder.model_builder import ModelBuilder

import argparse
import logging
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time


logging.basicConfig(
    stream=sys.stdout, level=logging.DEBUG,
    format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger()


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('model_name', type=str)
argument_parser.add_argument('number_of_iterations', type=int)
argument_parser.add_argument('ucb_worker_count', type=int)
argument_parser.add_argument('ucb_strength', type=int)
argument_parser.add_argument('ugtsa_worker_count', type=int)
argument_parser.add_argument('ugtsa_strength', type=int)
args = argument_parser.parse_args()


def random_game_state(game_state: GameState):
    for _ in range(random.randint(0, 40)):
        game_state.apply_move(random.randint(0, game_state.move_count() - 1))
    return game_state

config = tf.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.allow_growth=True

with tf.device('/gpu:0'):
    with tf.Session(config=config) as session:
        # load model
        try:
            saver = tf.train.import_meta_graph(
                'models/{model_name}/{model_name}.meta'
                .format(model_name=args.model_name))
            saver.restore(
                session, tf.train.latest_checkpoint(
                    'models/{}'.format(args.model_name)))
            logger.info('Restored model.')
        except Exception as e:
            logger.warning('Failed to restore model: {}.'.format(e))

            model_builder = ModelBuilder(
                variable_scope='model',
                player_count=2,
                worker_count=args.ugtsa_worker_count,
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
            model_builder.build()
            counter = tf.Variable(0, dtype=tf.int32, name='counter')
            increment_counter = tf.assign(counter, counter + 1)
            tf.add_to_collection('counter', counter)
            tf.add_to_collection('increment_counter', increment_counter)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

        # improve
        for _ in range(args.number_of_iterations):
            # initial game state
            # game_state = random_game_state(GameState(7))
            game_state = GameState(7)

            # get counter
            counter = session.run(tf.get_collection('counter')[0])

            # get model
            empty_statistic_model = session.run(
                tf.get_collection('model/empty_statistic/model_initializer')[0])
            move_rate_model = session.run(
                tf.get_collection('model/move_rate/model_initializer')[0])
            game_state_as_update_model = session.run(
                tf.get_collection('model/game_state_as_update/model_initializer')[0])
            updated_statistic_model = session.run(
                tf.get_collection('model/updated_statistic/model_initializer')[0])
            updated_update_model = session.run(
                tf.get_collection('model/updated_update/model_initializer')[0])

            # calculate ucb move rates
            ucb_begin = time.time()
            ucb_algorithm = UcbAlgorithm(
                game_state=deepcopy(game_state),
                worker_count=args.ucb_worker_count,
                grow_factor=5,
                exploration_factor=np.sqrt(2))
            for _ in range(args.ucb_strength):
                ucb_algorithm.improve()
            ucb_end = time.time()
            logger.info('{}: UCB took {}'.format(counter, ucb_end - ucb_begin))

            # calculate ugtsa move rates
            ugtsa_begin = time.time()
            ugtsa_algorithm = UgtsaAlgorithm(
                game_state=deepcopy(game_state),
                worker_count=args.ugtsa_worker_count,
                grow_factor=5,
                session=session,
                variable_scope='model',
                training=True,
                empty_statistic_model=empty_statistic_model,
                move_rate_model=move_rate_model,
                game_state_as_update_model=game_state_as_update_model,
                updated_statistic_model=updated_statistic_model,
                updated_update_model=updated_update_model)
            for _ in range(args.ugtsa_strength):
                ugtsa_algorithm.improve()
            ugtsa_end = time.time()
            logger.info('{}: UGTSA took {}'.format(counter, ugtsa_end - ugtsa_begin))

            # calculate gradients
            gradients_begin = time.time()
            results = session.run(
                fetches=[
                    tf.get_collection('model/cost_function/predicted_move_rates_gradient')[0],
                    tf.get_collection('model/cost_function/empty_statistic_model_gradient')[0],
                    tf.get_collection('model/cost_function/move_rate_model_gradient')[0],
                    tf.get_collection('model/cost_function/game_state_as_update_model_gradient')[0],
                    tf.get_collection('model/cost_function/updated_statistic_model_gradient')[0],
                    tf.get_collection('model/cost_function/updated_update_model_gradient')[0]],
                feed_dict={
                    tf.get_collection('model/cost_function/predicted_move_rates')[0]:
                        [ugtsa_algorithm.value(move_rate)
                         for move_rate in ugtsa_algorithm.move_rates()],
                    tf.get_collection('model/cost_function/real_move_rates')[0]:
                        [ucb_algorithm.value(move_rate)
                         for move_rate in ucb_algorithm.move_rates()],
                    tf.get_collection('model/cost_function/empty_statistic_model')[0]:
                        empty_statistic_model,
                    tf.get_collection('model/cost_function/move_rate_model')[0]:
                        move_rate_model,
                    tf.get_collection('model/cost_function/game_state_as_update_model')[0]:
                        game_state_as_update_model,
                    tf.get_collection('model/cost_function/updated_statistic_model')[0]:
                        updated_statistic_model,
                    tf.get_collection('model/cost_function/updated_update_model')[0]:
                        updated_update_model,
                })
            gradients = ugtsa_algorithm.computation_graph.gradients(
                xs=[ugtsa_algorithm.empty_statistic_model,
                    ugtsa_algorithm.move_rate_model,
                    ugtsa_algorithm.game_state_as_update_model,
                    ugtsa_algorithm.updated_statistic_model,
                    ugtsa_algorithm.updated_update_model],
                y_grads={
                    ugtsa_algorithm.empty_statistic_model: results[1],
                    ugtsa_algorithm.move_rate_model: results[2],
                    ugtsa_algorithm.game_state_as_update_model: results[3],
                    ugtsa_algorithm.updated_statistic_model: results[4],
                    ugtsa_algorithm.updated_update_model: results[5],
                    **{
                        move_rate: move_rate_gradient
                        for move_rate, move_rate_gradient in zip(
                            ugtsa_algorithm.move_rates(), results[0])
                    }
                })
            gradients_end = time.time()
            logger.info('{}: gradients took {}'.format(counter, gradients_end - gradients_begin))

            # update gradients
            empty_statistic_model += 0.001 * gradients[0]
            move_rate_model += 0.001 * gradients[1]
            game_state_as_update_model += 0.001 * gradients[2]
            updated_statistic_model += 0.001 * gradients[3]
            updated_update_model += 0.001 * gradients[4]

            # update graph
            session.run(
                [
                    tf.get_collection('increment_counter')[0],
                    tf.get_collection('model/empty_statistic/model_initializer_setter')[0],
                    tf.get_collection('model/move_rate/model_initializer_setter')[0],
                    tf.get_collection('model/game_state_as_update/model_initializer_setter')[0],
                    tf.get_collection('model/updated_statistic/model_initializer_setter')[0],
                    tf.get_collection('model/updated_update/model_initializer_setter')[0]
                ],
                {
                    tf.get_collection(
                        'model/empty_statistic/model_initializer_setter_input')[0]:
                            empty_statistic_model,
                    tf.get_collection(
                        'model/move_rate/model_initializer_setter_input')[0]:
                            move_rate_model,
                    tf.get_collection(
                        'model/game_state_as_update/model_initializer_setter_input')[0]:
                            game_state_as_update_model,
                    tf.get_collection(
                        'model/updated_statistic/model_initializer_setter_input')[0]:
                            updated_statistic_model,
                    tf.get_collection(
                        'model/updated_update/model_initializer_setter_input')[0]:
                            updated_update_model,
                })

            # store graph
            try:
                os.mkdir('models/{}'.format(args.model_name))
            except:
                pass
            saver.save(session, 'models/{model_name}/{model_name}'.format(
                model_name=args.model_name))

            # debug
            print([x.number_of_visits for x in ucb_algorithm.tree[:20]])
            print([x.number_of_visits for x in ugtsa_algorithm.tree[:20]])
            print(gradients)
            print([empty_statistic_model, move_rate_model, game_state_as_update_model,
                   updated_statistic_model, updated_update_model])
