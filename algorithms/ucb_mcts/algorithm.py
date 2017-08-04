from algorithms.generalized_mcts import algorithm
from games.game.game_state import GameState
from recordclass import recordclass

import numpy as np


class Algorithm(algorithm.Algorithm):
    Statistic = recordclass('Statistic', 'w n')
    Update = np.ndarray
    Rate = np.ndarray

    def __init__(self, game_state: GameState, number_of_workers: int, grow_factor: int, exploration_factor: int):
        super().__init__(game_state, number_of_workers, grow_factor)
        self.exploration_factor = exploration_factor

    def _empty_statistic(self, game_state: [GameState]) -> [Statistic]:
        return [Algorithm.Statistic(w=np.zeros(game_state[0].player_count), n=0)
                for _ in game_state]

    def _move_rate(self, parent_statistic: [Statistic], child_statistic: [Statistic]) -> [Rate]:
        return [cs.w / (cs.n + 0.1) + self.exploration_factor * np.sqrt(np.log(ps.n) / (cs.n + 0.1))
                for (ps, cs) in zip(parent_statistic, child_statistic)]

    def _game_state_as_update(self, game_state: [GameState]) -> [Update]:
        return [self._random_playout_payoff(gs) for gs in game_state]

    def _updated_statistic(self, statistic: [Statistic], updates: [[Update]]) -> [Statistic]:
        result = []

        for s, ps in zip(statistic, updates):
            ns = Algorithm.Statistic(w=np.array(s.w, copy=True), n=s.n + len(ps))
            for w in np.argmax(ps, 1):
                ns.w[w] += 1
            result += [ns]

        return result

    def _updated_update(self, update: [Update], statistic: [Statistic]) -> [Update]:
        return update

    def _run_batch(self) -> None:
        pass

    def value(self, rate: Rate) -> np.ndarray:
        return rate
