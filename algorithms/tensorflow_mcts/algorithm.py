from algorithms.generalized_mcts import algorithm
from algorithms.tensorflow_mcts.computation_graph import ComputationGraph
from games.game.game_state import GameState

import numpy as np
import tensorflow as tf


class Algorithm(algorithm.Algorithm):
    Statistic = int
    Update = int
    Rate = int

    def __init__(
            self, game_state: GameState, number_of_workers: int,
            grow_factor: int,
            session: tf.Session, variable_scope: str, training: bool,
            empty_statistic_model: np.ndarray,
            move_rate_model: np.ndarray,
            game_state_as_update_model: np.ndarray,
            updated_statistic_model: np.ndarray,
            updated_update_model: np.ndarray):
        super().__init__(game_state, number_of_workers, grow_factor)

        self.session = session
        self.variable_scope = variable_scope

        self.computation_graph = ComputationGraph(self.session)

        # training
        self.training = self.computation_graph.matrix(np.array(training))

        # empty statistic
        self.empty_statistic_model = \
            self.computation_graph.matrix(empty_statistic_model)
        self.empty_statistic_seed_size = self.session.run(
            tf.get_collection('{}/empty_statistic/seed_size'
                              .format(self.variable_scope))[0])
        self.empty_statistic_transformation = \
            self.__transformation(
                name='empty_statistic',
                inputs=['game_state_board',
                        'game_state_statistic'])

        # move rate
        self.move_rate_model = \
            self.computation_graph.matrix(move_rate_model)
        self.move_rate_seed_size = self.session.run(
            tf.get_collection('{}/move_rate/seed_size'
                              .format(self.variable_scope))[0])
        self.move_rate_transformation = \
            self.__transformation(
                name='move_rate',
                inputs=['parent_statistic',
                        'child_statistic'])

        # game state as update
        self.game_state_as_update_model = \
            self.computation_graph.matrix(game_state_as_update_model)
        self.game_state_as_update_seed_size = self.session.run(
            tf.get_collection('{}/game_state_as_update/seed_size'
                              .format(self.variable_scope))[0])
        self.game_state_as_update_transformation = \
            self.__transformation(
                name='game_state_as_update',
                inputs=['update_statistic'])

        # updated statistic
        self.updated_statistic_model = \
            self.computation_graph.matrix(updated_statistic_model)
        self.updated_statistic_seed_size = self.session.run(
            tf.get_collection('{}/updated_statistic/seed_size'
                              .format(self.variable_scope))[0])
        self.updated_statistic_transformation = \
            self.__transformation(
                name='updated_statistic',
                inputs=['statistic',
                        'update_count',
                        'updates'])

        # updated update
        self.updated_update_model = \
            self.computation_graph.matrix(updated_update_model)
        self.updated_update_seed_size = self.session.run(
            tf.get_collection('{}/updated_update/seed_size'
                              .format(self.variable_scope))[0])
        self.updated_update_transformation = \
            self.__transformation(
                name='updated_update',
                inputs=['update',
                        'statistic'])

    def __transformation(
            self, name: str, inputs: [str]) -> int:
        return self.computation_graph.transformation(
            batch_inputs=[
                tf.get_collection('{}/{}/{}'.format(
                    self.variable_scope, name, batch_input_name))[0]
                for batch_input_name in ['model', 'seed']] + [
                    tf.get_collection(
                        '{}/settings/training'
                        .format(self.variable_scope))[0]],
            batch_input_gradients=[
                tf.get_collection('{}/{}/{}_gradient'.format(
                    self.variable_scope, name, batch_input_name))[0]
                for batch_input_name in ['model', 'seed']],
            inputs=[
                tf.get_collection('{}/{}/{}'.format(
                    self.variable_scope, name, input))[0]
                    for input in inputs],
            input_gradients=[
                tf.get_collection('{}/{}/{}'.format(
                    self.variable_scope, name, input))[0]
                    for input in inputs],
            output=tf.get_collection(
                '{}/{}/output'.format(self.variable_scope, name)),
            output_gradient=tf.get_collection(
                '{}/{}/output_gradient'.format(self.variable_scope, name)))

    def _game_state_statistic(self, game_state: GameState):
        return game_state.random_playout_payoff()  # TODO: make abstract

    def _update_statistic(self, game_state: GameState):
        return game_state.random_playout_payoff()  # TODO: make abstract

    def _empty_statistic(self, game_state: GameState) -> Statistic:
        self.computation_graph.matrix(game_state.as_matrix())
        self.computation_graph.matrix(self._game_state_statistic(game_state))

        return self.computation_graph.transformation_run(
            transformation=self.empty_statistic_transformation,
            inputs=[self.computation_graph.matrix(game_state.as_matrix()),
                    self.computation_graph.matrix(
                        self._game_state_statistic(game_state))])

    def _move_rate(
            self, parent_statistic: Statistic, child_statistic: Statistic) \
            -> Rate:
        return self.computation_graph.transformation_run(
            transformation=self.move_rate_transformation,
            inputs=[parent_statistic, child_statistic])

    def _game_state_as_update(self, game_state: GameState) -> Update:
        return self.computation_graph.transformation_run(
            transformation=self.game_state_as_update_transformation,
            inputs=[self.computation_graph.matrix(
                self._update_statistic(game_state))])

    def _updated_statistic(self, statistic: Statistic, updates: [Update]) \
            -> Statistic:
        return self.computation_graph.transformation_run(
            transformation=self.updated_statistic_transformation,
            inputs=[statistic,
                    self.computation_graph.matrix(np.array(len(updates))),
                    updates])

    def _updated_update(self, update: Update, statistic: Statistic) \
            -> Update:
        return self.computation_graph.transformation_run(
            transformation=self.updated_update_transformation,
            inputs=[update, statistic])

    def _run_batch(self):
        batch_inputs = {
            transformation: [
                model,
                self.computation_graph.matrix(
                    np.random.randint(np.iinfo(np.int64).max, size=seed_size,
                                      dtype=np.int64)),
                self.training]
            for transformation, model, seed_size in [
                (self.empty_statistic_transformation,
                 self.empty_statistic_model,
                 self.empty_statistic_seed_size),
                (self.move_rate_transformation,
                 self.move_rate_model,
                 self.move_rate_seed_size),
                (self.game_state_as_update_transformation,
                 self.game_state_as_update_model,
                 self.game_state_as_update_seed_size),
                (self.updated_statistic_transformation,
                 self.updated_statistic_model,
                 self.updated_statistic_seed_size),
                (self.updated_update_transformation,
                 self.updated_update_model,
                 self.updated_update_seed_size),
            ]
        }

        self.computation_graph.run_batch(batch_inputs)

    def value(self, rate: Rate) -> np.ndarray:
        return self.computation_graph.value(rate)
