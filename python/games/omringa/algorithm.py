from algorithms.computation_graph_mcts import algorithm
from games.omringa.game_state import GameState
from computation_graphs.computation_graph.computation_graph import \
    ComputationGraph

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
            game_state, worker_count, grow_factor, removed_root_moves,
            computation_graph, empty_statistic, move_rate,
            game_state_as_update, updated_statistic, updated_update)

    def _game_state_statistic(self, game_state: GameState):
        return self.game_state.get_bets()

    def _update_statistic(self, game_state: GameState):
        return self.game_state.random_playout_payoff()
