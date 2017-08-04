from algorithms.generalized_mcts import algorithm
from algorithms.tensorflow_mcts.computation_graph import ComputationGraph
from games.game.game_state import GameState

import numpy as np
import tensorflow as tf


class Algorithm(algorithm.Algorithm):
    Statistic = int
    Update = int
    Rate = int

    def __init__(self, game_state: GameState, number_of_workers: int,
                 grow_factor: int, prefix: str, session: tf.Session,
                 empty_statistic_model,
                 move_rate_model,
                 game_state_as_update_model,
                 updated_statistic_model,
                 updated_update_model):
        self.prefix = prefix
        self.session = session

        self.computation_graph = ComputationGraph(self.session)

        # empty statistic
        self.empty_statistic_model = \
            self.computation_graph.matrix(empty_statistic_model)
        self.empty_statistic_transformation = \
            self.__transformation(
                name='empty_statistic',
                inputs=[
                    'game_state_board',
                    'game_state_statistic',
                ],
                model_matrix=self.empty_statistic_model)

        # move rate
        self.move_rate_model = \
            self.computation_graph.matrix(move_rate_model)
        self.move_rate_transformation = \
            self.__transformation(
                name='move_rate',
                inputs=[
                    'parent_statistic',
                    'child_statistic',
                ],
                model_matrix=self.move_rate_model)

        # game state as update
        self.game_state_as_update_model = \
            self.computation_graph.matrix(game_state_as_update_model)
        self.game_state_as_update_transformation = \
            self.__transformation(
                name='game_state_as_update',
                inputs=[
                    'update_statistic',
                ],
                model_matrix=self.game_state_as_update_model)

        # updated statistic
        self.updated_statistic_model = \
            self.computation_graph.matrix(updated_statistic_model)
        self.updated_statistic_transformation = \
            self.__transformation(
                name='updated_statistic',
                inputs=[
                    'statistic',
                    'update_count', # TODO: is it ok?
                    'updates', # TODO: one input is a list of inputs or a single input
                ],
                model_matrix=self.updated_statistic_model)

        # updated update
        self.updated_update_model = \
            self.computation_graph.matrix(updated_update_model)
        self.updated_update_transformation = \
            self.__transformation(
                name='updated_update',
                inputs=[
                    'update',
                    'statistic',
                ],
                model_matrix=self.updated_update_model)

        super().__init__(game_state, number_of_workers, grow_factor)

    def __transformation(self, name: str, inputs: [tf.Tensor], model_matrix: int) -> int:
        return self.computation_graph.transformation(
            model=tf.get_collection(
                '{}/{}/model'.format(self.prefix, name))[0],
            model_gradient=tf.get_collection(
                '{}/{}/model_gradient'.format(
                    self.prefix, name))[0],
            inputs=[tf.get_collection('{}/{}/{}'.format(
                self.prefix, name, input))[0] for input in inputs],
            input_gradients=[tf.get_collection('{}/{}/{}'.format(
                self.prefix, name, input))[0] for input in inputs],
            output=tf.get_collection(
                '{}/{}/output'.format(self.prefix, name)),
            output_gradient=tf.get_collection(
                '{}/{}/output_gradient'.format(self.prefix, name)),
            model_matrix=model_matrix)

    def _game_state_statistic(self, game_state: GameState):
        return game_state.random_playout_payoff()

    def _update_statistic(self, game_state: GameState):
        return game_state.random_playout_payoff()

    def _empty_statistic(self, game_state: [GameState]) -> [Statistic]:
        game_state_board = [
            self.computation_graph.matrix(gs.as_matrix())
            for gs in game_state]
        game_state_statistic = [
            self.computation_graph.matrix(self._game_state_statistic(gs))
            for gs in game_state]

        return [
            self.computation_graph.node(
                transformation=self.empty_statistic_transformation,
                inputs=[gsb, gss])
            for gsb, gss in zip(game_state_board, game_state_statistic)]

    def _move_rate(self, parent_statistic: [Statistic],
                   child_statistic: [Statistic]) -> [Rate]:
        return [
            self.computation_graph.node(
                transformation=self.move_rate_transformation,
                inputs=[ps, cs])
            for ps, cs in zip(parent_statistic, child_statistic)]

    def _game_state_as_update(self, game_state: [GameState]) -> [Update]:
        update_statistic = [
            self.computation_graph.matrix(self._update_statistic(gs))
            for gs in game_state]

        return [
            self.computation_graph.node(
                transformation=self.game_state_as_update_transformation,
                inputs=[us])
            for us in update_statistic]

    def _updated_statistic(self, statistic: [Statistic], updates: [[Update]]) \
            -> [Statistic]:
        update_count = [
            self.computation_graph.matrix(np.array(len(us)))
            for us in updates]
        return [
            self.computation_graph.node(
                transformation=self.updated_statistic_transformation,
                inputs=[s, uc, us])
            for s, uc, us in zip(statistic, update_count, updates)]

    def _updated_update(self, update: [Update], statistic: [Statistic]) \
            -> [Update]:
        return [
            self.computation_graph.node(
                transformation=self.updated_update_transformation,
                inputs=[u, s])
            for u, s in zip(update, statistic)]

    def _run_batch(self):
        self.computation_graph.run_batch()

    def value(self, rate: Rate) -> np.ndarray:
        return self.computation_graph.value(rate)
