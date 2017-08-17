from algorithms.generalized_mcts import algorithm
from computation_graphs.computation_graph.computation_graph \
    import ComputationGraph
from games.game.game_state import GameState

import numpy as np


class Algorithm(algorithm.Algorithm):
    Statistic = ComputationGraph.Node
    Rate = ComputationGraph.Node
    Update = ComputationGraph.Node
    Transformation = ComputationGraph.Transformation

    def __init__(
            self, game_state: GameState, worker_count: int, grow_factor: int,
            removed_root_moves: [int],
            computation_graph: ComputationGraph,
            empty_statistic: Transformation,
            move_rate: Transformation,
            game_state_as_update: Transformation,
            updated_statistic: Transformation,
            updated_update: Transformation):
        super().__init__(
            game_state, worker_count, grow_factor, removed_root_moves)

        self.computation_graph = computation_graph
        self.empty_statistic = empty_statistic
        self.move_rate = move_rate
        self.game_state_as_update = game_state_as_update
        self.updated_statistic = updated_statistic
        self.updated_update = updated_update

    def _game_state_statistic(self, game_state: GameState):
        raise NotImplementedError

    def _update_statistic(self, game_state: GameState):
        raise NotImplementedError

    def _empty_statistic(self, game_state: [GameState]) -> [Statistic]:
        return self.computation_graph.transformation_run(
            transformation=self.empty_statistic,
            inputs=[self.computation_graph.matrix(game_state.as_matrix()),
                    self.computation_graph.matrix(
                        self._game_state_statistic(game_state))])

    def _move_rate(
            self, parent_statistic: Statistic, child_statistic: Statistic) \
            -> Rate:
        return self.computation_graph.transformation_run(
            transformation=self.move_rate,
            inputs=[parent_statistic, child_statistic])

    def _game_state_as_update(self, game_state: GameState) -> Update:
        return self.computation_graph.transformation_run(
            transformation=self.game_state_as_update,
            inputs=[self.computation_graph.matrix(
                self._update_statistic(game_state))])

    def _updated_statistic(self, statistic: Statistic, updates: [Update]) \
            -> Statistic:
        return self.computation_graph.transformation_run(
            transformation=self.updated_statistic,
            inputs=[statistic,
                    self.computation_graph.matrix(np.array(len(updates))),
                    updates])

    def _updated_update(self, update: Update, statistic: Statistic) -> Update:
        return self.computation_graph.transformation_run(
            transformation=self.updated_update,
            inputs=[update, statistic])

    def _run_batch(self) -> None:
        self.computation_graph.run_batch()

    def value(self, rate: Rate) -> np.ndarray:
        return self.computation_graph.value(rate)
