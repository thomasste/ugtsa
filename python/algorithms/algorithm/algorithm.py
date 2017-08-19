from games.game.game_state import GameState

import math
import numpy as np


class Algorithm(object):
    Rate = object

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def move_rates(self) -> [Rate]:
        raise NotImplementedError

    def value(self, rate: Rate) -> np.ndarray:
        raise NotImplementedError

    def best_move(self) -> GameState.Move:
        assert not self.game_state.is_final()

        best_move = None
        best_rate = -math.inf
        for move, rates in enumerate(self.move_rates()):
            rates = self.value(rates)
            if best_rate < rates[self.game_state.player]:
                best_move = move
                best_rate = rates[self.game_state.player]

        return best_move
