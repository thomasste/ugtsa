from copy import deepcopy

import numpy as np
import random


class GameState(object):
    Move = int
    Payoff = np.ndarray

    def __init__(self):
        self.player = None
        self.player_count = None

    def move_count(self) -> int:
        raise NotImplementedError

    def get_move(self, index: int):
        raise NotImplementedError

    def apply_move(self, index: int) -> int:
        raise NotImplementedError

    def undo_move(self) -> int:
        raise NotImplementedError

    def is_final(self) -> bool:
        raise NotImplementedError

    def payoff(self) -> np.ndarray:
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def random_playout_payoff(self) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def random_game_state(cls, game_state):
        game_state = deepcopy(game_state)

        counter = 0
        while not game_state.is_final():
            game_state.apply_move(
                random.randint(0, game_state.move_count() - 1))
            counter += 1

        for _ in range(random.randint(0, counter)):
            game_state.undo_move()

        return game_state
