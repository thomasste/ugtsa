from algorithms.generalized_mcts import algorithm
from games.game.game_state import GameState
from recordclass import recordclass

import numpy as np


class Algorithm(algorithm.Algorithm):
    Statistic = recordclass('Statistic', 'w n')
    Update = np.ndarray
    Rate = np.ndarray

    def __init__(
            self, game_state: GameState, number_of_workers: int,
            grow_factor: int, exploration_factor: int):
        super().__init__(game_state, number_of_workers, grow_factor)

        self.exploration_factor = exploration_factor

    def _empty_statistic(self, game_state: GameState) -> Statistic:
        return Algorithm.Statistic(w=np.zeros(game_state.player_count), n=0)

    def _move_rate(
            self, parent_statistic: Statistic, child_statistic: Statistic) \
            -> Rate:
        return child_statistic.w / (child_statistic.n + 0.1) + \
               self.exploration_factor * \
               np.sqrt(np.log(parent_statistic.n) / (child_statistic.n + 0.1))

    def _game_state_as_update(self, game_state: GameState) -> Update:
        return game_state.random_playout_payoff()

    def _updated_statistic(
            self, statistic: Statistic, updates: [Update]) -> Statistic:
        new_statistic = Algorithm.Statistic(
            w=np.array(statistic.w, copy=True),
            n=statistic.n + len(updates))

        for w in np.argmax(updates, 1):
            new_statistic.w[w] += 1

        return new_statistic

    def _updated_update(self, update: Update, statistic: Statistic) -> Update:
        return update

    def _run_batch(self) -> None:
        pass

    def value(self, rate: Rate) -> np.ndarray:
        return rate
